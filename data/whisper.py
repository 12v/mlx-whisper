import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from params import sample_rate
from utils import device

pretrained_model = "openai/whisper-small"

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

    return output.input_features.to(torch.float32), output.attention_mask


def get_text_tensors(labels):
    tokenizer_output = tokenizer(labels, padding=True)
    input_label_tensor = torch.tensor(tokenizer_output.input_ids)[:, :-1]
    input_label_mask = torch.tensor(tokenizer_output.attention_mask)[:, :-1]

    output_label_tensor = torch.tensor(tokenizer_output.input_ids)[:, 1:]
    output_label_mask = torch.tensor(tokenizer_output.attention_mask)[:, 1:]

    return input_label_tensor, input_label_mask, output_label_tensor, output_label_mask


def collate_fn(batch):
    audios = [item[0] for item in batch]
    audio_input_features, audio_attention_mask = extract_audio_features(
        audios, sample_rate
    )

    labels = [item[1] for item in batch]
    input_label_tensor, input_label_mask, output_label_tensor, output_label_mask = (
        get_text_tensors(labels)
    )

    return (
        audio_input_features,
        audio_attention_mask,
        input_label_tensor,
        input_label_mask,
        output_label_tensor,
        output_label_mask,
    )
