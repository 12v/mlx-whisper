import json
import os
import wave
import xml.etree.ElementTree as ET

import numpy as np
from torch.utils.data import Dataset

from data.whisper import extract_audio_features, get_text_tensors
from params import sample_rate

script_dir = os.path.dirname(os.path.abspath(__file__))

speaker_change_token = "<|startoflm|>"
speaker_change_id = 50360


def get_wav_files():
    ami_dir = os.path.join(script_dir, "..", "ami", "amicorpus")

    wav_files = []

    for root, _, files in os.walk(ami_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.abspath(os.path.join(root, file)))

    return wav_files


def get_word_files():
    resources_dir = os.path.join(script_dir, "..", "ami", "ami_public_manual_1.6.2")
    words_dir = os.path.join(resources_dir, "words")

    transcripts = []

    for root, _, files in os.walk(words_dir):
        for file in files:
            if file.endswith(".xml"):
                transcripts.append(os.path.abspath(os.path.join(root, file)))

    return transcripts


def get_data_dict():
    wav_files = get_wav_files()
    word_files = get_word_files()

    data_dict = {}

    for wav_file in wav_files:
        meeting_id = wav_file.split("/")[-1].split(".")[0]
        if meeting_id not in data_dict:
            data_dict[meeting_id] = {"wav_file": wav_file, "word_files": []}
        data_dict[meeting_id]["wav_file"] = wav_file

    for word_file in word_files:
        meeting_id = word_file.split("/")[-1].split(".")[0]
        if meeting_id not in data_dict:
            continue

        data_dict[meeting_id]["word_files"].append(word_file)

    for meeting_id in list(data_dict.keys()):
        if len(data_dict[meeting_id]["word_files"]) == 0:
            del data_dict[meeting_id]

    return data_dict


def extract_words(word_files):
    words = []
    for word_file in word_files:
        with open(word_file, "r") as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for child in root:
                if child.tag == "w":
                    id = child.attrib["{http://nite.sourceforge.net/}id"]
                    speaker = id.split(".")[1]
                    if "starttime" not in child.attrib or "endtime" not in child.attrib:
                        continue

                    start_time = child.attrib["starttime"]
                    end_time = child.attrib["endtime"]
                    text = child.text
                    is_punctuation = child.attrib.get("punc", "false") == "true"
                    is_truncated = child.attrib.get("trunc", "false") == "true"
                    start_time = float(start_time)
                    end_time = float(end_time)
                    words.append(
                        {
                            "speaker": speaker,
                            "start_time": start_time,
                            "end_time": end_time,
                            "text": text,
                            "is_punctuation": is_punctuation,
                            "is_truncated": is_truncated,
                        }
                    )
    return words


def create_text_from_words(words):
    text = ""
    sorted_words = sorted(words, key=lambda x: x["start_time"])

    latest_speaker = None
    last_word = None
    for word in sorted_words:
        if (
            last_word
            and last_word["speaker"] != word["speaker"]
            and word["is_punctuation"]
        ):
            continue

        if word["text"].lower() in ["um", "uh"]:
            continue

        if word["is_truncated"]:
            continue

        if latest_speaker != word["speaker"] and latest_speaker is not None:
            text += "</" + latest_speaker + ">"

        if latest_speaker != word["speaker"] or latest_speaker is None:
            text += "<" + word["speaker"] + ">"

        text += (" " if not word["is_punctuation"] else "") + word["text"]

        latest_speaker = word["speaker"]
        last_word = word

    if latest_speaker is not None:
        text += "</" + latest_speaker + ">"

    return text


def create_text_from_words_with_speaker_change(words):
    text = ""
    sorted_words = sorted(words, key=lambda x: x["start_time"])

    latest_speaker = None
    last_word = None
    for word in sorted_words:
        if (
            last_word
            and last_word["speaker"] != word["speaker"]
            and word["is_punctuation"]
        ):
            continue

        if word["text"].lower() in ["um", "uh"]:
            continue

        if word["is_truncated"]:
            continue

        if latest_speaker != word["speaker"] and latest_speaker is not None:
            text += speaker_change_token

        text += (" " if not word["is_punctuation"] else "") + word["text"]

        latest_speaker = word["speaker"]
        last_word = word

    return text


def chunk_transcripts(wav_file_path, word_files):
    chunks = []

    words = extract_words(word_files)

    with wave.open(wav_file_path, "rb") as wav_file:
        length = wav_file.getnframes() / wav_file.getframerate()

    for i in range(0, int(length), 30):
        start = i
        end = min(i + 30, length)
        chunk_length = end - start
        if chunk_length < 30:
            continue

        chunk_words = [
            word
            for word in words
            if word["start_time"] >= start and word["end_time"] <= end
        ]

        text = create_text_from_words_with_speaker_change(chunk_words)

        chunks.append(
            {"wav_file": wav_file_path, "start": start, "end": end, "text": text}
        )

    return chunks


dump_path = os.path.join(script_dir, "ami_dataset.json")
if not os.path.exists(dump_path):
    data_dict = get_data_dict()
    dataset = []

    for meeting_data in data_dict.values():
        wav_file = meeting_data["wav_file"]
        word_files = meeting_data["word_files"]
        chunks = chunk_transcripts(wav_file, word_files)
        dataset.extend(chunks)

    with open(dump_path, "w") as f:
        json.dump(dataset, f)

else:
    with open(dump_path, "r") as f:
        dataset = json.load(f)


class AMIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        start = item["start"]
        end = item["end"]
        text = item["text"]

        wav_file = item["wav_file"]
        with wave.open(wav_file, "rb") as wav_file:
            wav_file.setpos(int(start * wav_file.getframerate()))
            audio = wav_file.readframes(int((end - start) * wav_file.getframerate()))

        audio = np.frombuffer(audio, dtype=np.int16)
        return audio, text


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
