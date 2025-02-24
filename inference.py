import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from audio import load_audio
from params import sample_rate
from utils import device

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-tiny", language="english"
)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="english")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(
    device
)

model.eval()
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None


def transcribe(audio_file):
    audio = load_audio(audio_file, sample_rate)

    features = feature_extractor(
        audio,
        return_tensors="pt",
        sampling_rate=sample_rate,
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        output = model.generate(**features)

    return tokenizer.decode(output[0])


if __name__ == "__main__":
    audio_file = "hello.wav"
    result = transcribe(audio_file)
    print(result)
