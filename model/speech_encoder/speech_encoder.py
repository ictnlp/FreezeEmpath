# This code is modified from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import torch
import torch.nn as nn


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        import whisper
        encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        replace_layer_norm(encoder)
        return encoder


class SenseVoiceSmallEncoder(nn.Module):
    def __init__(self, kwargs, model):
        super().__init__()
        self.kwargs = kwargs
        self.model = model

    @classmethod
    def load(cls, model_config):
        from funasr import AutoModel
        model = AutoModel(model=model_config.speech_encoder, device="cuda", disable_update=True)
        return cls(model.kwargs, model.model)
    
    def inference(self, speech, speech_lengths):
        speech, speech_lengths = self.kwargs["frontend"](speech.float().cpu(), speech_lengths)
        speech = speech.to(torch.float32).cuda()
        speech_lengths = speech_lengths.to(torch.int32).cuda()
        language = self.kwargs.get("language", "auto")
        language_query = self.model.embed(
            torch.LongTensor([[self.model.lid_dict[language] if language in self.model.lid_dict else 0]]).to(
                speech.device
            )
        ).repeat(speech.size(0), 1, 1)

        textnorm = "withitn"
        textnorm_query = self.model.embed(
            torch.LongTensor([[self.model.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.model.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.model.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # # c. Passed the encoder result and the beam search
        # ctc_logits = self.model.ctc.log_softmax(encoder_out)
        # if self.kwargs.get("ban_emo_unk", False):
        #     ctc_logits[:, :, self.model.emo_dict["unk"]] = -float("inf")

        # b, n, d = encoder_out.size()
        # for i in range(b):
        #     x = ctc_logits[i, : encoder_out_lens[i].item(), :]
        #     yseq = x.argmax(dim=-1)
        #     yseq = torch.unique_consecutive(yseq, dim=-1)

        #     mask = yseq != self.model.blank_id
        #     token_int = yseq[mask].tolist()

        #     # Change integer-ids to tokens
        #     text = self.kwargs["tokenizer"].decode(token_int)
        #     print("ASR Output:", text)
        
        encoder_out = encoder_out.to(dtype=torch.bfloat16)

        return encoder_out, encoder_out_lens