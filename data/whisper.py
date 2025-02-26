import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from utils import device

pretrained_model = "openai/whisper-base"

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
    output = feature_extractor(
        audios,
        return_tensors="pt",
        sampling_rate=sample_rate,
        return_attention_mask=True,
        padding="max_length",
    )

    return output.input_features, output.attention_mask


def get_text_tensors(labels):
    tokenizer_output = tokenizer(labels, padding=True)
    input_label_tensor = torch.tensor(tokenizer_output.input_ids)[:, :-1]
    input_label_mask = torch.tensor(tokenizer_output.attention_mask)[:, :-1]

    output_label_tensor = torch.tensor(tokenizer_output.input_ids)[:, 1:]
    output_label_mask = torch.tensor(tokenizer_output.attention_mask)[:, 1:]

    return input_label_tensor, input_label_mask, output_label_tensor, output_label_mask
