import aiohttp
import datasets
import torch

from data.whisper import collate_fn

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
