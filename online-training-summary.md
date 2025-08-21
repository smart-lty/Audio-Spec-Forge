# SpecForge Online Training 逻辑分析总结

## 概述

SpecForge 是一个基于Eagle3的投机性解码(Speculative Decoding)框架，支持在线训练模式。在线训练是指在训练过程中实时从目标模型(Target Model)获取隐藏状态，用于训练草稿模型(Draft Model)的方式。

## 核心组件

### 1. 训练入口脚本
- **主脚本**: `scripts/train_eagle3_online.py`
- **运行示例**: `examples/run_llama3_eagle3_online.sh`

### 2. 核心模型类
- **OnlineEagle3Model**: `specforge/core/eagle3.py:37-260`
  - 实现了在线训练的核心逻辑
  - 支持Test-Time Training (TTT)技术
  - 动态提取目标模型隐藏状态

## 在线训练核心流程

### 1. 模型初始化阶段

#### 目标模型(Target Model)
```python
# 单GPU模式
target_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.target_model_path,
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
).eval().cuda()

# 多GPU分布式模式
target_model = AutoDistributedTargetModel.from_pretrained(
    pretrained_model_name_or_path=args.target_model_path,
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
    device="cuda",
).eval()
```

#### 草稿模型(Draft Model)
```python
# 从配置文件创建或从检查点恢复
draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
if draft_model_last_checkpoint:
    draft_model = AutoEagle3DraftModel.from_pretrained(draft_model_last_checkpoint)
else:
    draft_model = AutoEagle3DraftModel.from_config(draft_model_config)

# 加载embedding并冻结
draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
draft_model.freeze_embedding()
```

### 2. 数据处理流程

#### 数据预处理
- **对话格式化**: 使用chat template处理ShareGPT格式对话数据
- **token化**: 将对话转换为token序列
- **损失掩码**: 仅对助手回复部分计算损失

```python
def preprocess_conversations(tokenizer, conversations, chat_template, max_length):
    # 应用chat template格式化对话
    conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # token化
    encoding = tokenizer(conversation, max_length=max_length, truncation=True)
    
    # 生成损失掩码，仅对助手回复计算损失
    loss_mask = torch.zeros(len(input_ids))
    # 使用正则表达式标记助手回复区域
    for match in re.finditer(assistant_pattern, conversation):
        # 标记助手回复的token位置
```

#### 词汇映射生成
- **统计token频率**: 分析训练数据中有效token的使用频率
- **生成映射表**: 创建目标词汇表到草稿词汇表的映射关系

```python
def generate_vocab_mapping_file(dataset, target_vocab_size, draft_vocab_size):
    # 统计有效token频率
    token_dict = Counter()
    for item in dataset:
        masked_ids = input_ids[loss_mask == 1]  # 仅统计需要预测的token
        
    # 选择最频繁的draft_vocab_size个token
    top_N = token_dict.most_common(draft_vocab_size)
    
    # 生成双向映射表 d2t 和 t2d
    return d2t, t2d
```

### 3. Test-Time Training (TTT) 核心逻辑

#### 隐藏状态提取
```python
def _prepare_data(self, input_ids, attention_mask, loss_mask):
    # 运行目标模型获取所有层的隐藏状态
    outputs = self.target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    
    # 提取3个辅助层的隐藏状态
    low_aux_layer = 1 + offset                    # 第1层
    mid_aux_layer = num_layers // 2 - 1 + offset  # 中间层
    last_aux_layer = num_layers - 4 + offset      # 倒数第4层
    
    # 拼接3个层的隐藏状态
    hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)
    
    return hidden_states, target_logits, loss_mask, input_ids
```

#### TTT训练循环
```python
def forward(self, input_ids, attention_mask, loss_mask):
    # 步骤1: 提取目标模型隐藏状态
    hidden_states, target, loss_mask, input_ids = self._prepare_data(...)
    
    # 步骤2: 投影隐藏状态维度 (3*hidden_size -> hidden_size)
    hidden_states = self.draft_model.project_hidden_states(hidden_states)
    
    # 步骤3-7: TTT循环训练
    for idx in range(self.length):  # 默认7轮
        # 步骤3.1: 嵌入输入token
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        
        # 步骤3.2: 运行草稿模型骨干网络
        hidden_states_out = self.draft_model.backbone(
            input_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        
        # 步骤3.3: 处理词汇表映射
        target_head = target[..., self.draft_model.t2d]  # 映射到草稿词汇表
        target_p = nn.Softmax(dim=2)(target_head)
        
        # 步骤3.4: 计算草稿模型输出logits
        logits = self.draft_model.compute_logits(hidden_states)
        
        # 步骤3.5: 计算KL散度损失
        out_logp = nn.LogSoftmax(dim=2)(logits)
        plogp = target_p * out_logp
        loss = -torch.sum(position_mask * plogp, 2).mean()
        
        # 步骤3.6: 记录准确率
        accuracy = (logits.argmax(-1) == target_p.argmax(-1)).sum() / loss_mask.sum()
        
        # 步骤3.7: 更新attention mask用于下一轮
        if not is_last:
            # 屏蔽未来位置的注意力
            attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min
```

### 4. 训练配置参数

#### 核心训练参数
- **ttt_length**: TTT展开长度 (默认7)
- **num_epochs**: 训练轮数 (默认10)
- **batch_size**: 批次大小 (默认1)
- **learning_rate**: 学习率 (默认1e-4)
- **max_length**: 最大序列长度 (默认2048)
- **warmup_ratio**: 预热比例 (默认0.02)

#### 损失权重策略
```python
# 对不同TTT步骤使用指数衰减权重
ploss_weight = [0.8**i for i in range(len(plosses))]
ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
```

### 5. 分布式训练支持

#### FSDP配置
```python
eagle3_model = FSDP(
    eagle3_model,
    use_orig_params=True,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    ignored_modules=[target_model],  # 目标模型不参与梯度分片
    process_group=get_dp_group(),
)
```

#### 优化器和调度器
```python
optimizer = torch.optim.AdamW(eagle3_model.parameters(), lr=args.learning_rate)
scheduler = CosineAnnealingWarmupLR(
    optimizer, 
    total_steps=total_steps, 
    warmup_steps=warmup_steps
)
```

## 在线训练vs离线训练对比

| 特性 | 在线训练 (OnlineEagle3Model) | 离线训练 (OfflineEagle3Model) |
|------|------------------------------|------------------------------|
| 隐藏状态获取 | 实时从目标模型提取 | 预先计算并存储 |
| 内存占用 | 需要加载完整目标模型 | 仅需要草稿模型 |
| 训练灵活性 | 可以处理任意新数据 | 仅限于预处理的数据 |
| 计算开销 | 每个batch需要运行目标模型 | 无需运行目标模型 |
| 适用场景 | 探索性训练，小规模数据 | 大规模生产训练 |

## 关键技术特点

### 1. 多层隐藏状态融合
- 提取目标模型的3个关键层: 第1层、中间层、倒数第4层
- 通过拼接形成更丰富的表示 (3*hidden_size)
- 投影到草稿模型的隐藏维度

### 2. 词汇表映射优化
- 基于训练数据频率选择最重要的token子集
- 减少草稿模型词汇表大小，提升训练效率
- 保持与目标模型的兼容性

### 3. Test-Time Training机制
- 模拟推理时的渐进式生成过程
- 7轮迭代训练，逐步提升预测能力
- 使用attention mask模拟因果关系

### 4. 损失设计
- 仅对助手回复部分计算损失，避免对系统提示过拟合
- KL散度损失确保与目标模型分布对齐
- 指数衰减权重平衡不同TTT步骤的重要性

## 总结

SpecForge的在线训练模式提供了一种灵活的草稿模型训练方案，通过实时提取目标模型的隐藏状态，使用TTT技术训练出能够有效加速推理的草稿模型。该方法在保持与目标模型高度一致性的同时，显著减少了推理时间，是投机性解码领域的重要技术贡献。