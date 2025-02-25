import random
from collections import defaultdict

import aiohttp
import datasets
import torch

from data.whisper import extract_audio_features, get_text_tensors
from params import sample_rate

train_dataset = datasets.load_dataset(
    "librispeech_asr",
    "clean",
    split="train.100",
    trust_remote_code=True,
    storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=18000)}},
)
test_dataset = datasets.load_dataset(
    "librispeech_asr",
    "clean",
    split="test",
    trust_remote_code=True,
    storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=18000)}},
)


def collate_fn(batch):
    audios = [item[0] for item in batch]
    audio_input_features, audio_attention_mask = extract_audio_features(
        audios, sample_rate
    )

    labels = [item[1] for item in batch]
    input_label_tensor, input_label_mask, output_label_tensor, output_label_mask = (
        get_text_tensors(labels)
    )

    return (
        audio_input_features,
        audio_attention_mask,
        input_label_tensor,
        input_label_mask,
        output_label_tensor,
        output_label_mask,
    )


def speaker_id_collate_fn(batch):
    audios = [item[0] for item in batch]
    relevant_audios = [item[1] for item in batch]
    irrelevant_audios = [item[2] for item in batch]

    audio_input_features, audio_attention_mask = extract_audio_features(
        audios, sample_rate
    )

    relevant_audio_input_features, relevant_audio_attention_mask = (
        extract_audio_features(relevant_audios, sample_rate)
    )

    irrelevant_audio_input_features, irrelevant_audio_attention_mask = (
        extract_audio_features(irrelevant_audios, sample_rate)
    )

    return (
        audio_input_features.permute(0, 2, 1),
        audio_attention_mask,
        relevant_audio_input_features.permute(0, 2, 1),
        relevant_audio_attention_mask,
        irrelevant_audio_input_features.permute(0, 2, 1),
        irrelevant_audio_attention_mask,
    )


def process_row(row):
    audio = row["audio"]["array"]
    text = row["text"]

    return audio, text


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return process_row(row)


class IterableLibriSpeechDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for row in self.dataset:
            yield process_row(row)


class SpeakerIdDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.speaker_ids = self.dataset.unique("speaker_id")

        self.speaker_id_to_index = defaultdict(list)
        for i, speaker_id in enumerate(self.dataset["speaker_id"]):
            self.speaker_id_to_index[speaker_id].append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        relevant_row = None
        irrelevant_row = None

        relevant_ids = self.speaker_id_to_index[row["speaker_id"]]
        while relevant_row is None:
            other_row = self.dataset[random.choice(relevant_ids)]
            if row["id"] != other_row["id"]:
                relevant_row = other_row

        while irrelevant_row is None:
            other_row = self.dataset[random.randint(0, len(self.dataset) - 1)]
            if row["speaker_id"] != other_row["speaker_id"]:
                irrelevant_row = other_row

        return (
            row["audio"]["array"],
            relevant_row["audio"]["array"],
            irrelevant_row["audio"]["array"],
        )


def get_dataloaders(batch_size):
    train = LibriSpeechDataset(train_dataset)
    test = LibriSpeechDataset(test_dataset)

    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        ),
        torch.utils.data.DataLoader(
            test, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        ),
    )
