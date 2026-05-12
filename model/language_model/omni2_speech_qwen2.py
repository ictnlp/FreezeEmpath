from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..omni2_speech_arch import Omni2SpeechMetaModel, Omni2SpeechMetaForCausalLM
from constants import IGNORE_INDEX


class Omni2SpeechQwen2Config(Qwen2Config):
    model_type = "omni2_speech_qwen2"


class Omni2SpeechQwen2Model(Omni2SpeechMetaModel, Qwen2Model):
    config_class = Omni2SpeechQwen2Config

    def __init__(self, config: Qwen2Config):
        super(Omni2SpeechQwen2Model, self).__init__(config)


class Omni2SpeechQwen2ForCausalLM(Qwen2ForCausalLM, Omni2SpeechMetaForCausalLM):
    config_class = Omni2SpeechQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Omni2SpeechQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.hidden2emotion = nn.Linear(config.hidden_size, 5, bias=False)
        self.emo_label_dict = {0: 'angry', 1: 'surprised', 2: 'happy', 3: 'sad', 4: 'neutral'}
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        add_ser_loss: bool = False, 
        emotion_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                emotion_features
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if not add_ser_loss:
            return output
        else:
            logits = output.logits
            total_loss = inputs_embeds.new_zeros(())
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            response_ce_loss = F.cross_entropy(shifted_logits[shifted_labels != IGNORE_INDEX],
                                    shifted_labels[shifted_labels != IGNORE_INDEX], reduction="mean")
            total_loss += response_ce_loss
            
            ser_logits = self.hidden2emotion(emotion_features)
            ser_loss = F.cross_entropy(ser_logits.view(-1, 5), emotion_labels.view(-1))
            total_loss += 0 * ser_loss
            
            return {"loss": total_loss}
        
        
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs

AutoConfig.register("omni2_speech_qwen2", Omni2SpeechQwen2Config)
AutoModelForCausalLM.register(Omni2SpeechQwen2Config, Omni2SpeechQwen2ForCausalLM)
