# coding=utf-8
# Copyright 2024 SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from specforge.core.eagle3 import OnlineEagle3Model
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


class AudioOnlineEagle3Model(OnlineEagle3Model):
    """
    Audio-enabled EAGLE-3 model for multimodal training with audio inputs.
    Extends OnlineEagle3Model to handle audio features alongside text tokens.
    """

    def __init__(self, target_model, draft_model: Eagle3DraftModel, length: int = 7):
        """
        Args:
            target_model: Qwen2-Audio target model that processes audio inputs
            draft_model: EAGLE-3 draft model to be trained
            length: TTT length for unrolling during training
        """
        super().__init__(target_model, draft_model, length)
    
    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract hidden states from Qwen2-Audio model with audio inputs.
        
        Returns:
            hidden_states: (batch, seq_len, 3*hidden_size) - concatenated hidden states
            target: (batch, seq_len, vocab_size) - target logits
            loss_mask: (batch, seq_len) - loss mask
            input_ids: (batch, seq_len) - padded input ids
            audio_embeds: (batch, seq_len, hidden_size) - merged text+audio embeddings
        """
        if device is None:
            device = input_ids.device
        
        # Forward pass through Qwen2-Audio model with audio inputs
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        
        # Extract aux hidden states
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1
        
        # Eagle3 uses 3 aux layers
        low_aux_layer = 1 + offset
        mid_aux_layer = num_layers // 2 - 1 + offset
        last_aux_layer = num_layers - 4 + offset
        
        hidden_states = torch.cat(
            (outputs.hidden_states[low_aux_layer], 
             outputs.hidden_states[mid_aux_layer], 
             outputs.hidden_states[last_aux_layer]), 
            dim=-1
        )
        
        # Get the merged embeddings from model output
        audio_embeds = outputs.inputs_embeds
        
        # Apply padding
        target = padding(outputs.logits, left=False)
        input_ids = padding(input_ids, left=False)
        audio_embeds = padding(audio_embeds, left=False)
        
        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None].to(device)
        
        return hidden_states, target, loss_mask, input_ids, audio_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for audio-enabled EAGLE-3 training with Qwen2-Audio.
        
        Args:
            input_ids: (batch, seq_len) - includes audio placeholder tokens
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            input_features: (batch, num_frames, num_mel_bins) - audio features
            feature_attention_mask: (batch, num_frames) - audio mask
            past_key_values: KV cache (not used in Eagle3)
            position_ids: (batch, seq_len)
            
        Returns:
            plosses: List of prediction losses
            vlosses: List of value losses (empty for Eagle3)
            acces: List of accuracy values
        """
        
        # Get data with audio embeddings
        hidden_states, target, loss_mask, input_ids, audio_embeds = self._prepare_data(
            input_ids, attention_mask, loss_mask, input_features, feature_attention_mask
        )
        
        # Rest of the forward pass is the same as OnlineEagle3Model
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        # Project hidden states
        hidden_states = self.draft_model.project_hidden_states(hidden_states)
        
        # Process position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
        )
        
        # Run TTT (Test-Time Training)
        plosses = []
        vlosses = []
        acces = []
        cache_hidden = [[], []]
        
        for idx in range(self.length):
            is_last = idx == self.length - 1
            
            # Use pre-computed audio-aware embeddings (contains both text and audio features)
            inputs_embeds = audio_embeds.to(hidden_states.dtype)
            
            # Run draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            
            # Handle vocab size mapping
            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                target_mask = self.draft_model.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.draft_model.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            
            # Update hidden states for next step
            hidden_states = hidden_states_out
            
            # Get logits
            logits = self.draft_model.compute_logits(hidden_states)
            logits = logits.float()
            
            # Calculate loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()

            # Record metrics
            plosses.append(loss)
            with torch.no_grad():
                acces.append(
                    (
                        (logits.argmax(-1) == target_p.argmax(-1))
                        * position_mask.squeeze(-1)
                    )
                    .sum()
                    .item()
                    / (loss_mask.sum().item() + 1e-6)
                )
            
            # Update for next iteration
            if not is_last:
                # For audio inputs, we keep using the same audio_embeds throughout
                # No need to update input_ids since we use audio_embeds directly
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)
                ind = torch.arange(seq_length, device=attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_length - idx]
                attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min
        
        return plosses, vlosses, acces