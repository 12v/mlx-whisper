# MLX Whisper

## Setup

1. Clone the repository
2. Install the required dependencies:
```python
pip install -r requirements.txt
```

3. Download the AMI Corpus data:
```bash
./data/download_ami_corpus.sh
```

4. Download and extract the AMI annotations from [here](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip)

## Usage

### Inference
```bash
python inference.py
```

### Overfitting
```bash
python overfit.py
```

### LibriSpeech Finetuning
```bash
python finetune_librispeech.py
```

### Diarisation Training
```bash
python train_diarisation.py
```