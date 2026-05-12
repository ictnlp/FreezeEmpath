import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from constants import IGNORE_INDEX


def lengths_to_attention_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return ~mask


class LLMSpeechGenerator(nn.Module):
    def __init__(self, config):
        super(LLMSpeechGenerator, self).__init__()
        self.model = Qwen2ForCausalLM(Qwen2Config(**config.speech_generator))
        self.tokenizer = AutoTokenizer.from_pretrained(config.tts_tokenizer)
        if getattr(config, 'speech_generator_proj_type', 'linear') == 'linear':
            self.input_proj = nn.Linear(config.hidden_size, self.model.config.hidden_size)
        elif getattr(config, 'speech_generator_proj_type', 'linear') == 'ffn':
            tts_ffn_dim_size = getattr(config, 'tts_ffn_dim_size', -1)
            ffn_dim = tts_ffn_dim_size if tts_ffn_dim_size != -1 else config.hidden_size * 2
            self.input_proj = nn.Sequential(
                nn.Linear(config.hidden_size, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, self.model.config.hidden_size)
            )
        self.stream_params = config.stream_params
        self.tts_max_length = config.tts_max_length
        self.num_sample_turns_s2s = config.num_sample_turns_s2s
        self.add_token_embeddings = config.add_token_embeddings
        self.add_gate_fusion = getattr(config, 'add_gate_fusion', False)
        if self.add_gate_fusion:
            self.gate = nn.Sequential(
                nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size),
                nn.Sigmoid()
            )

    def find_label_segments(self, labels):
        mask = (labels != IGNORE_INDEX)
        int_mask = mask.to(torch.int8)
        diff_mask = torch.diff(
            int_mask,
            prepend=torch.tensor([0], device=int_mask.device, dtype=int_mask.dtype),
            append=torch.tensor([0], device=int_mask.device, dtype=int_mask.dtype),
        )
        starts = (diff_mask == 1).nonzero().squeeze(-1)
        ends = (diff_mask == -1).nonzero().squeeze(-1) - 1
        return starts, ends

    def fusion(self, rep, emb):
        if self.add_gate_fusion:
            gate = self.gate(torch.cat([rep, emb], dim=-1))
            return rep * gate + emb * (1 - gate)
        else:
            return rep + emb

    def get_llm_output(self, tgt_reps, labels, tgt_units):
        tgt_label_reps = []
        tgt_units_training = []
        cur_index = 0
        for tgt_rep, label in zip(tgt_reps, labels):
            starts, ends = self.find_label_segments(label)
            cur_label_reps = [self.input_proj(tgt_rep[s-1:e-1]) for s, e in zip(starts, ends)]
            if self.add_token_embeddings:
                cur_label_embeddings = [self.model.get_input_embeddings()(label[s:e]) for s, e in zip(starts, ends)]
                cur_label_reps = [self.fusion(rep, emb) for rep, emb in zip(cur_label_reps, cur_label_embeddings)]
            cur_units = tgt_units[cur_index:cur_index+len(cur_label_reps)]
            cur_index += len(cur_label_reps)
            if self.num_sample_turns_s2s > 0:
                selected_index = random.sample(range(len(cur_label_reps)), min(len(cur_label_reps), self.num_sample_turns_s2s))
                cur_label_reps = [cur_label_reps[i] for i in selected_index]
                cur_units = [cur_units[i] for i in selected_index]
            tgt_label_reps.extend(cur_label_reps)
            tgt_units_training.extend(cur_units)
        return tgt_label_reps, tgt_units_training

    def get_unit_embeddings(self, tgt_unit):
        unit_str = "".join([f"<{u}>" for u in tgt_unit])
        unit_ids = self.tokenizer(unit_str, padding=False, return_tensors="pt")["input_ids"][0].to(tgt_unit.device)
        unit_embeds = self.model.get_input_embeddings()(unit_ids)
        return unit_ids, unit_embeds

    def get_stream_input_and_labels(self, tgt_label_reps, tgt_units):
        N, M = eval(self.stream_params)
        device = tgt_units[0].device

        sep_id = torch.LongTensor([self.tokenizer.convert_tokens_to_ids("<sep>")]).to(device)
        eos_id = torch.LongTensor([self.tokenizer.eos_token_id]).to(device)
        sep_emb = self.model.get_input_embeddings()(sep_id)
        eos_emb = self.model.get_input_embeddings()(eos_id)

        input_embeds_list = []
        labels_list = []
        for tgt_label_rep, tgt_unit in zip(tgt_label_reps, tgt_units):
            tgt_unit = tgt_unit[tgt_unit != IGNORE_INDEX]
            tgt_unit, tgt_unit_embed = self.get_unit_embeddings(tgt_unit)

            _tgt_label_rep = torch.cat([tgt_label_rep, sep_emb], dim=0)
            _tgt_unit = torch.cat([tgt_unit, eos_id], dim=0)
            _tgt_unit_embed = torch.cat([tgt_unit_embed, eos_emb], dim=0)
            
            tgt_label_rep_chunks = [_tgt_label_rep[i:i+N] for i in range(0, len(_tgt_label_rep), N)]
            tgt_unit_chunks = [_tgt_unit[i:i+M] for i in range(0, len(_tgt_unit), M)]
            tgt_unit_embed_chunks = [_tgt_unit_embed[i:i+M] for i in range(0, len(_tgt_unit_embed), M)]

            if len(tgt_unit_chunks) < len(tgt_label_rep_chunks):
                input_embeds = torch.cat([_tgt_label_rep, _tgt_unit_embed], dim=0)
                labels = torch.cat([torch.full((_tgt_label_rep.size(0),), IGNORE_INDEX, dtype=torch.long, device=device), _tgt_unit], dim=0)
                input_embeds_list.append(input_embeds)
                labels_list.append(labels)
                continue

            input_embeds_seq = []
            labels_seq = []
            for i, t_chunk in enumerate(tgt_label_rep_chunks):
                input_embeds_seq.append(t_chunk)
                labels_seq.append(torch.full((t_chunk.size(0),), IGNORE_INDEX, dtype=torch.long, device=device))
                if i == len(tgt_label_rep_chunks) - 1:
                    u_chunk = _tgt_unit[i * M:]
                    u_emb_chunk = _tgt_unit_embed[i * M:]
                else:
                    u_chunk = tgt_unit_chunks[i]
                    u_emb_chunk = tgt_unit_embed_chunks[i]
                input_embeds_seq.append(u_emb_chunk)
                labels_seq.append(u_chunk)
            
            input_embeds = torch.cat(input_embeds_seq, dim=0)
            labels = torch.cat(labels_seq, dim=0)
            input_embeds_list.append(input_embeds)
            labels_list.append(labels)
        
        padded_input_embeds = nn.utils.rnn.pad_sequence(
            input_embeds_list, 
            batch_first=True, 
            padding_value=0
        )[:, :self.tts_max_length, :]
        padded_labels = nn.utils.rnn.pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )[:, :self.tts_max_length]
        attention_mask = lengths_to_attention_mask(torch.LongTensor([x.size(0) for x in input_embeds_list]))

        return padded_input_embeds, attention_mask.to(device), padded_labels
                
    def forward(self, tgt_reps, labels, tgt_units):
        tgt_label_reps, tgt_units = self.get_llm_output(tgt_reps, labels, tgt_units)
        input_embeds, attention_mask, labels = self.get_stream_input_and_labels(tgt_label_reps, tgt_units)
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate_units(self, tts_inputs, new_hidden_states, new_tokens, is_finished=False):
        # only for batch size = 1
        new_hidden_states = self.input_proj(new_hidden_states)
        if self.add_token_embeddings:
            new_token_embeddings = self.model.get_input_embeddings()(new_tokens)
            new_hidden_states = self.fusion(new_hidden_states, new_token_embeddings)
        if tts_inputs is not None:
            tts_inputs = torch.cat([tts_inputs, new_hidden_states], dim=0)
        else:
            tts_inputs = new_hidden_states
        if is_finished:
            device = tts_inputs.device
            sep_id = torch.LongTensor([self.tokenizer.convert_tokens_to_ids("<sep>")]).to(device)
            sep_emb = self.model.get_input_embeddings()(sep_id)
            tts_inputs = torch.cat([tts_inputs, sep_emb], dim=0)

        _, M = eval(self.stream_params)
        max_new_tokens = M if not is_finished else 1024
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=tts_inputs.unsqueeze(0),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0]
        generated_units = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_tokens_embeds = self.model.get_input_embeddings()(generated_tokens)
        tts_inputs = torch.cat([tts_inputs, generated_tokens_embeds], dim=0)
        return tts_inputs, generated_units