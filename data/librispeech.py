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
    # streaming=True,
)
test_dataset = datasets.load_dataset(
    "librispeech_asr",
    "clean",
    split="test",
    trust_remote_code=True,
    storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=18000)}},
    # streaming=True,
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
