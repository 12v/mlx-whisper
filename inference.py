import torch

from data.audio import load_audio
from data.whisper import extract_audio_features, model, tokenizer
from params import sample_rate
from utils import device


def transcribe(audio):
    model.eval()

    features = extract_audio_features(audio, sample_rate)

    with torch.no_grad():
        output = model.generate(**features.to(device))

    return tokenizer.decode(output[0])


if __name__ == "__main__":
    audio = load_audio("hello.wav", sample_rate)
    result = transcribe(audio)
    print(result)
