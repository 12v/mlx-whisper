import torch

from data.audio import load_audio
from data.whisper import extract_audio_features, model, tokenizer
from params import sample_rate
from utils import device


def transcribe(audio):
    model.eval()

    input_features, attention_mask = extract_audio_features(audio, sample_rate)

    with torch.no_grad():
        input_token_ids = tokenizer.encode(["a"])[:-2]
        output_token_ids = input_token_ids[1:]

        for _ in range(100):
            output = model(
                input_features.to(device),
                attention_mask=attention_mask.to(device),
                decoder_input_ids=torch.tensor(input_token_ids).unsqueeze(0).to(device),
            )
            one_batch = output.logits.squeeze(0)
            element = one_batch[len(output_token_ids)]
            max = torch.argmax(element)
            item = max.item()
            token = tokenizer.decode(item)
            input_token_ids.append(item)
            output_token_ids.append(item)
            if token == "<|endoftext|>":
                break

    return tokenizer.decode(output_token_ids, skip_special_tokens=False)


if __name__ == "__main__":
    audio = load_audio("hello.wav", sample_rate)
    result = transcribe(audio)
    print(result)
