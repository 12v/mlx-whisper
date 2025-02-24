import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from data.librispeech import (
    LibriSpeechDataset,
    collate_fn,
    test_dataset,
    train_dataset,
)
from data.whisper import model, tokenizer
from params import sample_rate
from utils import conditional_autocast, device

model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

num_epochs = 10


batch_size = 128 if device.type == "cuda" else 4
num_workers = 0 if device.type == "cuda" else 0
persistent_workers = True if num_workers > 0 else False

train_dataset = LibriSpeechDataset(train_dataset, sample_rate)
test_dataset = LibriSpeechDataset(test_dataset, sample_rate)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True
)

scaler = GradScaler(device.type)

for epoch in range(num_epochs):
    loader = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
    losses = []
    model.train()
    optimizer.zero_grad()
    for (
        audio_features,
        audio_attention_mask,
        input_labels,
        input_labels_mask,
        output_labels,
        output_labels_mask,
    ) in loader:
        with conditional_autocast():
            output = model(
                input_features=audio_features.to(device),
                attention_mask=audio_attention_mask.to(device),
                decoder_input_ids=input_labels.to(device),
            )
            logits = output.logits
            loss = criterion(logits.permute(0, 2, 1), output_labels.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        loader.set_postfix(loss=sum(losses) / len(losses))
    print(f"Epoch {epoch} loss: {sum(losses) / len(losses)}")
