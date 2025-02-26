import torch
from tqdm import tqdm

from data.ami import AMIDataset, collate_fn, dataset
from data.whisper import model
from utils import device

batch_size = 32 if device.type == "cuda" else 4
num_workers = 2 if device.type == "cuda" else 0
persistent_workers = True if num_workers > 0 else False
initial_lr = 0.0001
num_epochs = 10

model.train()

train_loader = torch.utils.data.DataLoader(
    AMIDataset(dataset), batch_size=batch_size, collate_fn=collate_fn, shuffle=True
)


optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
criterion = torch.nn.CrossEntropyLoss(reduction="none")

for epoch in range(num_epochs):
    loader = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
    losses = []
    optimizer.zero_grad()
    for (
        audio_features,
        audio_attention_mask,
        input_labels,
        input_labels_mask,
        output_labels,
        output_labels_mask,
    ) in loader:
        model.train()
        output = model(
            input_features=audio_features.to(device),
            attention_mask=audio_attention_mask.to(device),
            decoder_input_ids=input_labels.to(device),
        )
        logits = output.logits
        loss = criterion(logits.permute(0, 2, 1), output_labels.to(device))
        loss = loss * output_labels_mask.to(device)
        loss = loss.sum() / output_labels_mask.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loader.set_postfix(loss=sum(losses) / len(losses))

    print(f"Epoch {epoch} loss: {sum(losses) / len(losses)}")
