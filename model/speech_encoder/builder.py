from .speech_encoder import WhisperWrappedEncoder, SenseVoiceSmallEncoder


def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)
    elif "sensevoice" in speech_encoder_type.lower():
        return SenseVoiceSmallEncoder.load(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
