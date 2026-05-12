from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
from .emotion_extractor.builder import build_emotion_extractor
from constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX, EMOTION_TOKEN_INDEX

import os
import numpy as np

class Omni2SpeechMetaModel:

    def __init__(self, config):
        super(Omni2SpeechMetaModel, self).__init__(config)
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
            self.emotion_extractor = build_emotion_extractor(config) 
            
    def get_speech_encoder(self):
        speech_encoder = getattr(self, "speech_encoder", None)
        return speech_encoder
    
    def get_speech_projector(self):
        speech_projector = getattr(self, "speech_projector", None)
        return speech_projector
    
    def get_emotion_extractor(self):
        emotion_extractor = getattr(self, 'emotion_extractor', None)
        return emotion_extractor
    
    def initialize_speech_modules(self, model_args):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            self.speech_encoder = build_speech_encoder(self.config)
        if self.get_speech_projector() is None:
            self.speech_projector = build_speech_projector(self.config)
        if self.get_emotion_extractor() is None:
            self.emotion_extractor = build_emotion_extractor(self.config)


class Omni2SpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().get_speech_projector()

    def get_emotion_extractor(self):
        return self.get_model().get_emotion_extractor()
     
    def extract_emotion_and_speech(self, speech, speech_lengths):
        emotion_extractor = self.get_emotion_extractor()
        speech_encoder = self.get_speech_encoder()

        x = speech.permute(0, 2, 1)
        x = F.gelu(speech_encoder.conv1(x))
        x = F.gelu(speech_encoder.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == speech_encoder.positional_embedding.shape, "incorrect audio shape"
        x = (x + speech_encoder.positional_embedding).to(x.dtype)
        hidden_states = []
        for block in speech_encoder.blocks:
            x = block(x)
            hidden_states.append(x)

        encoder_outs = speech_encoder.ln_post(x)
        speech_lengths = (speech_lengths + 1) // 2
        
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f"Unknown speech projector type: {speech_projector_type}")
        
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        emotion_features = emotion_extractor(torch.stack(hidden_states, dim=1))
        emotion_features = [emotion_features[i, :] for i in range(len(emotion_features))]

        return speech_features, emotion_features     
    
    
    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()

        if (speech_encoder is None) or (speech is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
        speech_features = []
        emotion_features = []
        total_speech = 0
        total_emotion = 0

        speech_features, emotion_features = self.extract_emotion_and_speech(speech, speech_lengths)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        cur_emotion_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech_or_emotion = (cur_input_ids == SPEECH_TOKEN_INDEX).sum() + (cur_input_ids == EMOTION_TOKEN_INDEX).sum()
            # assert num_speech==1, num_speech
            total_speech += (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            total_emotion += (cur_input_ids == EMOTION_TOKEN_INDEX).sum()
            speech_token_indices = [-1] + sorted(torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + torch.where(cur_input_ids == EMOTION_TOKEN_INDEX)[0].tolist()) + [cur_input_ids.shape[0]]
            # assert 0, speech_token_indices
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech_or_emotion + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if i < num_speech_or_emotion:
                    if cur_input_ids[speech_token_indices[i+1]] == SPEECH_TOKEN_INDEX:
                        # assert 0
                        cur_speech_features = speech_features[cur_speech_idx]
                        cur_speech_idx += 1
                    else:
                        assert cur_input_ids[speech_token_indices[i+1]] == EMOTION_TOKEN_INDEX
                        if emotion_features[cur_emotion_idx].dim() == 1:
                            assert 0, "wrong dim of emotion feature"
                            cur_speech_features = emotion_features[cur_emotion_idx].unsqueeze(0)
                        # cur_speech_features = emotion_features[cur_emotion_idx]
                        else:
                            assert emotion_features[cur_emotion_idx].dim() == 2
                            cur_speech_features = emotion_features[cur_emotion_idx]
                        cur_emotion_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    # print(cur_new_input_embeds[0].size(), cur_speech_features.size())
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            
        assert cur_speech_idx == total_speech
        assert cur_emotion_idx == total_emotion

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if emotion_features:
            emotion_features = torch.cat(emotion_features, dim=0)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, emotion_features