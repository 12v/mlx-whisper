import os

import torch
from torch.nn import functional as F
from tqdm import tqdm

from data.ami import AMIDataset, speaker_change_token
from data.ami import dataset as ami_dataset
from data.librispeech import LibriSpeechDataset
from data.librispeech import train_dataset as librispeech_dataset
from data.whisper import collate_fn, model, pretrained_model, tokenizer
from inference import transcribe
from utils import DummyWandb, conditional_autocast, device

if torch.cuda.is_available():
    import wandb
else:
    wandb = DummyWandb()

script_dir = os.path.dirname(os.path.abspath(__file__))

batch_size = 8 if device.type == "cuda" else 4
num_workers = 2 if device.type == "cuda" else 0
persistent_workers = True if num_workers > 0 else False
initial_lr = 1e-7
num_epochs = 20
lambda_speaker = 10
# model.load_state_dict(
#     torch.load(
#         os.path.join(script_dir, "weights/model_0_combined.pt"),
#         map_location=device,
#     )
# )

model.train()

ami_dataset = AMIDataset(ami_dataset)
librispeech_dataset = LibriSpeechDataset(librispeech_dataset)
joint_dataset = torch.utils.data.ConcatDataset([ami_dataset, librispeech_dataset])
train_loader = torch.utils.data.DataLoader(
    joint_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
)

audio, text = joint_dataset[4]


def test():
    print("ground truth")
    print("-" * 100)
    print(text)
    print("-" * 100)
    print("transcribed")
    print("-" * 100)
    with conditional_autocast():
        print(transcribe(audio))
    print("-" * 100)


test()

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
criterion = torch.nn.CrossEntropyLoss(reduction="none")


wandb.init(
    project="whisper-diarisation",
    config={
        "pretrained_model": pretrained_model,
        "batch_size": batch_size,
        "lr": initial_lr,
        "num_epochs": num_epochs,
        "lambda_speaker": lambda_speaker,
    },
)
wandb.log({"epoch": 0, "loss": 0})

speaker_change_token_id = tokenizer.encode(
    speaker_change_token, add_special_tokens=False
)[0]

scaler = torch.amp.GradScaler()


def combined_loss(logits, output_labels, output_labels_mask):
    speaker_change_mask = (output_labels == speaker_change_token_id).float()
    transcription_mask = output_labels_mask * (1 - speaker_change_mask)

    transcription_loss = criterion(logits.permute(0, 2, 1), output_labels.to(device))
    transcription_loss = transcription_loss * transcription_mask.to(device)
    transcription_loss = transcription_loss.sum() / transcription_mask.sum()

    speaker_change_logits = logits[:, :, speaker_change_token_id]
    speaker_change_loss = F.binary_cross_entropy_with_logits(
        speaker_change_logits, speaker_change_mask.to(device)
    )

    return (
        transcription_loss + lambda_speaker * speaker_change_loss,
        transcription_loss,
        speaker_change_loss,
    )


for epoch in range(num_epochs):
    loader = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
    losses = []
    transcription_losses = []
    speaker_change_losses = []

    for i, (
        audio_features,
        audio_attention_mask,
        input_labels,
        input_labels_mask,
        output_labels,
        output_labels_mask,
    ) in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        with conditional_autocast():
            output = model(
                input_features=audio_features.to(device),
                attention_mask=audio_attention_mask.to(device),
                decoder_input_ids=input_labels.to(device),
            )

            loss, transcription_loss, speaker_change_loss = combined_loss(
                output.logits, output_labels, output_labels_mask
            )

        speaker_change_loss.backward()
        optimizer.step()
        losses.append(loss.item())
        transcription_losses.append(transcription_loss.item())
        speaker_change_losses.append(speaker_change_loss.item())
        loader.set_postfix(
            loss=sum(losses) / len(losses),
            t_loss=sum(transcription_losses) / len(transcription_losses),
            s_loss=sum(speaker_change_losses) / len(speaker_change_losses),
        )
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "transcription_loss": transcription_loss.item(),
                "speaker_change_loss": speaker_change_loss.item(),
            }
        )

        if i % 10 == 0:
            test()

    os.makedirs(os.path.join(script_dir, "weights"), exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(script_dir, f"weights/model_{epoch}.pt"),
    )
    wandb.save(os.path.join(script_dir, f"weights/model_{epoch}.pt"))
    print(f"Epoch {epoch} loss: {sum(losses) / len(losses)}")
    wandb.log(
        {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
        }
    )
wandb.finish()
