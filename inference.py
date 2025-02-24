import torch

from data.audio import load_audio
from data.whisper import extract_audio_features, model, tokenizer
from params import sample_rate
from utils import device

model.eval()


def transcribe(audio_file):
    audio = load_audio(audio_file, sample_rate)

    features = extract_audio_features(audio, sample_rate)

    with torch.no_grad():
        output = model.generate(**features.to(device))

    return tokenizer.decode(output[0])


if __name__ == "__main__":
    audio_file = "hello.wav"
    result = transcribe(audio_file)
    print(result)
