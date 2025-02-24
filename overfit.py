import torch

from data.audio import load_audio
from data.whisper import extract_audio_features, model, tokenizer
from params import sample_rate
from utils import device

model.train()


audio_file = "hello.wav"
text = " Hello, my name's Izaak."

audio = load_audio(audio_file, sample_rate)
tokenizer_output = tokenizer(text)
labels = tokenizer_output.input_ids

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    features = extract_audio_features(audio, sample_rate).to(device)

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
    features = extract_audio_features(audio, sample_rate).to(device)
    output = model.generate(**features)
    print(tokenizer.decode(output[0]))
