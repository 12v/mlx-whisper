import torch

from data.audio import load_audio
from data.whisper import extract_audio_features, model, tokenizer
from params import sample_rate
from utils import device


def transcribe(audio):
    model.eval()

    input_features, attention_mask = extract_audio_features(audio, sample_rate)

    with torch.no_grad():
        output = model.generate(
            input_features.to(device), attention_mask=attention_mask.to(device)
        )

    return tokenizer.decode(output[0])


if __name__ == "__main__":
    audio = load_audio("hello.wav", sample_rate)
    result = transcribe(audio)
    print(result)
