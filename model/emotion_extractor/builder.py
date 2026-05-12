from .emotion_extractor import EmotionExtractor


def build_emotion_extractor(config):
    extractor_type = getattr(config, 'emotion_extrator_type', 'linear')
    if extractor_type == 'linear':
        return EmotionExtractor(config)

    raise ValueError(f'Unknown projector type: {extractor_type}')
