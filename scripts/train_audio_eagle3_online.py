import argparse
import hashlib
import os

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    build_multimodal_eagle3_dataset,
    multimodal_collate_fn,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import (
    get_last_checkpoint,
    print_with_rank,
    rank_0_priority,
    validate_wandb_args,
)
from specforge.modeling.target.qwen2_audio import Qwen2AudioForConditionalGeneration
from specforge.core.eagle3_multimodal import AudioOnlineEagle3Model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )

    # resume
    parser.add_argument("--resume", action="store_true")

    # wandb wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)

    args = parser.parse_args()

    return parser, args


def init_wandb(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)


def wandb_log_if_initialized(log_dict):
    if dist.get_rank() == 0 and wandb.run is not None:
        wandb.log(log_dict)


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank(f"Initialized distributed environment")

    # Validate wandb arguments
    validate_wandb_args(parser, args)

    if args.wandb and dist.get_rank() == 0:
        init_wandb(args)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    target_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.target_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
    
    print_with_rank(f"Initialized target model")
    # load model with resume
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(draft_model_last_checkpoint)
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(draft_model_config)
            .cuda()
            .to(torch.bfloat16)
        )
    print_with_rank(f"Initialized draft model")

    # build dataloaders
    processor = AutoProcessor.from_pretrained(args.target_model_path)

    # convert to dataloader
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.target_model_path}"  # Processor may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    
    with rank_0_priority():
        train_eagle3_dataset = build_multimodal_eagle3_dataset(
            data_path=args.train_data_path,  # Direct JSONL path
            processor=processor,
            max_length=args.max_length,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
    # Create distributed sampler and dataloader for multimodal data
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    train_sampler = DistributedSampler(
        train_eagle3_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_eagle3_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        collate_fn=multimodal_collate_fn,
    )
    print_with_rank(f"Initialized train dataloader")

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank(f"Loaded vocab mapping")

    if args.eval_data_path is not None:
        eval_eagle3_dataset = build_multimodal_eagle3_dataset(
            data_path=args.eval_data_path,  # Direct JSONL path
            processor=processor,
            max_length=args.max_length,
        )
        # Create distributed sampler and dataloader for eval multimodal data
        eval_sampler = dist.DistributedSampler(
            eval_eagle3_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False
        )
        eval_dataloader = DataLoader(
            eval_eagle3_dataset,
            batch_size=args.batch_size,
            sampler=eval_sampler,
            num_workers=4,
            pin_memory=False,
            collate_fn=multimodal_collate_fn,
        )
        print_with_rank(f"Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    eagle3_model = AudioOnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
        length=args.ttt_length,
    )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized Eagle3 FSDP model")

    # build other components
    optimizer = torch.optim.AdamW(eagle3_model.parameters(), lr=args.learning_rate)
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )
    print_with_rank(f"Initialized optimizer and scheduler")

    # resume
    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)

            optimizer.load_state_dict(state["optimizer_state_dict"])
            print_on_rank0("Successfully loaded optimizer state_dict.")

            scheduler.load_state_dict(state["scheduler_state_dict"])
            print_on_rank0("Successfully loaded scheduler state_dict.")

            start_epoch = state["epoch"] + 1
            print_on_rank0(f"Resuming from epoch {start_epoch}")
        else:
            print_on_rank0(
                f"Warning: Checkpoint directory {draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
            )

    dist.barrier()

    # start running
    print_on_rank0(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.module.length)]

        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
                input_features=data["input_features"].cuda(),  # ← Add audio features
                feature_attention_mask=data["feature_attention_mask"].cuda(),  # ← Add audio attention
            )

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            ploss.backward()
            optimizer.step()
            scheduler.step()

            logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb_log_if_initialized(logdict)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = acc_i / dist.get_world_size()
            acc_i = acc_i.item()
            wandb_log_if_initialized({f"train/epochacc_{i}": acc_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = loss_i / dist.get_world_size()
            loss_i = loss_i.item()
            wandb_log_if_initialized({f"train/epochploss_{i}": loss_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )

        # run evaluation
        if args.eval_data_path is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                    input_features=data["input_features"].cuda(),
                    feature_attention_mask=data["feature_attention_mask"].cuda(),
                )
                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            for i in range(len(epoch_acces)):
                acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = acc_i / dist.get_world_size()
                acc_i = acc_i.item()

                wandb_log_if_initialized({f"eval/epochacc_{i}": acc_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = loss_i / dist.get_world_size()
                loss_i = loss_i.item()

                wandb_log_if_initialized({f"eval/epochploss_{i}": loss_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )

        if epoch % args.save_interval == 0:
            # Save the model
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()

            with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
                model_state_dict = eagle3_model.state_dict()
                state_to_save = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                draft_model_state_dict = {
                    k.replace("draft_model.", ""): v
                    for k, v in model_state_dict.items()
                    if "draft_model." in k and "embed" not in k.lower()
                }

                if dist.get_rank() == 0:
                    torch.save(
                        state_to_save,
                        os.path.join(epoch_output_dir, "training_state.pt"),
                    )
                    print_on_rank0(
                        f"Saved full training state to {epoch_output_dir}/training_state.pt"
                    )
                    draft_model.save_pretrained(
                        epoch_output_dir,
                        state_dict=draft_model_state_dict,
                    )
                    print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                dist.barrier()

    destroy_distributed()


if __name__ == "__main__":
    main()
