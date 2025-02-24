from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from utils import device

pretrained_model = "openai/whisper-tiny"

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    pretrained_model, language="english"
)
tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model, language="english", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(pretrained_model).to(device)

model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None


def extract_audio_features(audios, sample_rate):
    return feature_extractor(
        audios,
        return_tensors="pt",
        sampling_rate=sample_rate,
        return_attention_mask=True,
        padding="max_length",
    )
