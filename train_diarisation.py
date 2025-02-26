import os

import torch
from tqdm import tqdm

from data.ami import AMIDataset, collate_fn, dataset
from data.whisper import model, pretrained_model
from inference import transcribe
from utils import DummyWandb, device

if torch.cuda.is_available():
    import wandb
else:
    wandb = DummyWandb()

script_dir = os.path.dirname(os.path.abspath(__file__))

batch_size = 16 if device.type == "cuda" else 4
num_workers = 2 if device.type == "cuda" else 0
persistent_workers = True if num_workers > 0 else False
initial_lr = 0.0000001
num_epochs = 10

model.load_state_dict(
    torch.load(os.path.join(script_dir, "weights/model_1_speaker_switch.pt"))
)

model.train()

train_dataset = AMIDataset(dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
)

audio, text = train_dataset[4]

print(transcribe(audio))
print(text)

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
criterion = torch.nn.CrossEntropyLoss(reduction="none")


wandb.init(
    project="whisper-diarisation",
    config={
        "pretrained_model": pretrained_model,
        "batch_size": batch_size,
        "initial_lr": initial_lr,
        "num_epochs": num_epochs,
    },
)
wandb.log({"epoch": 0, "loss": 0})

for epoch in range(num_epochs):
    loader = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
    losses = []
    optimizer.zero_grad()
    for i, (
        audio_features,
        audio_attention_mask,
        input_labels,
        input_labels_mask,
        output_labels,
        output_labels_mask,
    ) in enumerate(loader):
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
        wandb.log({"epoch": epoch, "loss": loss.item()})

        if i % 10 == 0:
            print(transcribe(audio))
            print(text)

    # create models directory if it doesn't exist
    os.makedirs(os.path.join(script_dir, "weights"), exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(script_dir, f"weights/model_{epoch}.pt"),
    )
    print(f"Epoch {epoch} loss: {sum(losses) / len(losses)}")
    wandb.log({"epoch": epoch, "loss": sum(losses) / len(losses)})
wandb.finish()
