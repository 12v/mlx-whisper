import torch
from tqdm import tqdm

from data.librispeech import (
    SpeakerIdDataset,
    speaker_id_collate_fn,
    test_dataset,
    train_dataset,
)
from model.speaker_id_encoder import Encoder
from utils import device

batch_size = 32 if device.type == "cuda" else 4
num_workers = 2 if device.type == "cuda" else 0
persistent_workers = True if num_workers > 0 else False
initial_lr = 0.001
num_epochs = 10
embedding_dim = 80
d_model = 10
margin = 0.3
dropout_rate = 0.2
num_layers = 2
num_heads = 2

model = Encoder(
    d_model=embedding_dim,
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout_rate=dropout_rate,
).to(device)

model.train()

train = SpeakerIdDataset(train_dataset)
test = SpeakerIdDataset(test_dataset)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, collate_fn=speaker_id_collate_fn, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, collate_fn=speaker_id_collate_fn, shuffle=False
)

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)


def compute_loss(
    model, audio_features, relevant_audio_features, irrelevant_audio_features
):
    audio_embeddings, _ = model(audio_features.to(device))
    relevant_audio_embeddings, _ = model(relevant_audio_features.to(device))
    irrelevant_audio_embeddings, _ = model(irrelevant_audio_features.to(device))

    relevant_similarity = torch.nn.functional.cosine_similarity(
        audio_embeddings, relevant_audio_embeddings
    )
    irrelevant_similarity = torch.nn.functional.cosine_similarity(
        audio_embeddings, irrelevant_audio_embeddings
    )

    relevant_distance = 1 - relevant_similarity
    irrelevant_distance = 1 - irrelevant_similarity

    unclamped_loss = relevant_distance - irrelevant_distance + margin

    loss = torch.max(torch.tensor(0.0), unclamped_loss).mean()

    return loss, unclamped_loss, relevant_distance, irrelevant_distance


for epoch in range(num_epochs):
    loader = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
    losses = []
    optimizer.zero_grad()
    for (
        audio_features,
        audio_attention_mask,
        relevant_audio_features,
        relevant_audio_attention_mask,
        irrelevant_audio_features,
        irrelevant_audio_attention_mask,
    ) in loader:
        model.train()

        loss, unclamped_loss, relevant_distance, irrelevant_distance = compute_loss(
            model, audio_features, relevant_audio_features, irrelevant_audio_features
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loader.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    print(f"Epoch {epoch} loss: {sum(losses) / len(losses)}")
