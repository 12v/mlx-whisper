import librosa


def load_audio(file_path, sample_rate):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio
