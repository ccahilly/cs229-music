# Investigating Audio Encoding Models' Effectiveness on Downstream Tasks

## Authors

Caroline Cahilly, Caleb Liu, Thomas Yim\
Department of Computer Science, Stanford University\
Contact: [ccahilly@stanford.edu](mailto\:ccahilly@stanford.edu), [calebliu@stanford.edu](mailto\:calebliu@stanford.edu), [yimt@stanford.edu](mailto\:yimt@stanford.edu)

---

## Overview

This project investigates the effectiveness of audio embedding models in downstream tasks like genre classification and music captioning. We compare three state-of-the-art audio embedding methods:

- **CLAP** (Contrastive Language-Audio Pretraining)
- **Wav2Vec 2.0**
- **MERT** (Acoustic Music Understanding Model)

We evaluate these embeddings' ability to capture rich, musically relevant information for both classification and text generation tasks.

---

## Motivation

Music evokes emotional and cognitive responses in people. From music recommendation systems to music therapy applications, robust audio embeddings can help machine learning systems understand and utilize musical features effectively.

Key challenges include:

- Capturing semantics like instrumentation, rhythm, and emotional tone.
- Generating embeddings that generalize across different musical styles.

---

## Datasets

### 1. **GTZAN Dataset** (Genre Classification)

- **1000** audio clips, each **30 seconds** long, spread across **10 genres**.
- Train-test split: **800/200**.

### 2. **MusicCaps Dataset** (Music Captioning)

- **5,360** examples: each pairs a **10-second song clip** with a **text description**.
- Train/val/test split: **70-10-20**.
- Descriptions include instrument details, emotional tone, and sequencing.

**Preprocessing Steps:**

- Resampling audio to match embedding model requirements:
  - CLAP: **48 kHz**
  - Wav2Vec 2.0: **24 kHz**
  - MERT: **24 kHz**
- Converting all audio to mono and normalizing amplitudes to [-1, 1].

---

## Methods

### Embedding Models

1. **CLAP**: Contrastive learning aligns audio embeddings with text descriptions. It optimizes embeddings using a contrastive loss function that pushes mismatched pairs apart.
2. **Wav2Vec 2.0**: Learns audio representations through self-supervised training and fine-tunes them with labeled data.
3. **MERT**: Combines acoustic and pitch-based embeddings using transformer layers optimized with classification and reconstruction losses.

### Genre Classification Pipeline

1. Extract embeddings.
2. Feed embeddings into an MLP classifier.
3. Optimize with **cross-entropy loss**.

### Music Captioning Pipeline

1. Extract embeddings.
2. Use T5 transformer to generate text captions.
3. Optimize with **cross-entropy loss**.

---

## Results

### Genre Classification Performance

| Embedding Model        | Train Accuracy (%) | Test Accuracy (%) |
| ---------------------- | ------------------ | ----------------- |
| Wav2Vec 2.0 (Frozen)   | 44.25              | 43.50             |
| CLAP (Frozen)          | 37.00              | 36.63             |
| MERT (Frozen)          | 72.12              | 66.50             |
| Wav2Vec 2.0 (Unfrozen) | 92.75              | 67.50             |
| CLAP (Unfrozen)        | 63.75              | 43.00             |
| MERT (Unfrozen)        | **100.00**         | **84.00**         |

- **Key Observations**:
  - MERT outperforms others, especially when fine-tuned (84% test accuracy).
  - CLAP struggles due to its general-purpose training, whereas MERT is music-specific.
  - Unfreezing embedding weights improves performance across models.

### PCA Analysis

PCA visualizations show clearer cluster separation in embeddings after fine-tuning, especially for MERT. This improved separation aligns with better classification performance.

---

## Conclusion

This project highlights the importance of model selection and fine-tuning for audio-related ML tasks. MERT emerges as the best performer, leveraging its music-specific pretraining.

**Future Work:**

- Investigating hybrid approaches combining multiple embeddings.
- Extending evaluation to tasks like emotion recognition and music synthesis.

---

## Repository Structure

```
.
├── data/                     # Dataset files and preprocessed data
├── models/                   # Pre-trained and fine-tuned embedding models
├── scripts/                  # Training and evaluation scripts
├── results/                  # Logs, visualizations, and output predictions
├── README.md                 # Project description and setup instructions
└── requirements.txt          # Required Python packages
```

---

## Setup

### Requirements

```
python>=3.8
pytorch>=1.10
torchaudio
transformers
scikit-learn
numpy
pandas
```

### Installation

```bash
git clone https://github.com/ccahilly/cs229-music.git
cd cs229-music
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python scripts/train.py --task genre_classification --model mert --finetune
```

### Evaluation

```bash
python scripts/evaluate.py --task genre_classification --model mert
```

---

## References

- Ding, Y. et al., "Audio Embedding Transfer Learning," 2023.
- Perera, M. et al., "Contrastive Learning for Audio Classification," 2024.
- GTZAN Dataset: [https://www.tensorflow.org/datasets/catalog/gtzan](https://www.tensorflow.org/datasets/catalog/gtzan)
- MusicCaps Dataset: [https://google-research.github.io/musiccaps/](https://google-research.github.io/musiccaps/)
