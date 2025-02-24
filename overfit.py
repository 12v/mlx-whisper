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
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-tiny", language="english", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(
    device
)

model.train()
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None


audio_file = "hello.wav"
text = " Hello, my name's Izaak."


audio = load_audio(audio_file, sample_rate)
tokenizer_output = tokenizer(text)
labels = tokenizer_output.input_ids


features = feature_extractor(
    audio,
    return_tensors="pt",
    sampling_rate=sample_rate,
    return_attention_mask=True,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    features = feature_extractor(
        audio,
        return_tensors="pt",
        sampling_rate=sample_rate,
        return_attention_mask=True,
    ).to(device)

    input_label_tensor = torch.tensor(labels[:-1]).to(device).unsqueeze(0)
    output_label_tensor = torch.tensor(labels[1:]).to(device).unsqueeze(0)

    output = model(**features, decoder_input_ids=input_label_tensor)

    logits = output.logits

    loss = criterion(logits.permute(0, 2, 1), output_label_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")


model.eval()
with torch.no_grad():
    features = feature_extractor(
        audio,
        return_tensors="pt",
        sampling_rate=sample_rate,
        return_attention_mask=True,
    ).to(device)
    output = model.generate(**features)
    print(tokenizer.decode(output[0]))
