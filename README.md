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
- Using embeddings that work on different downstream applications.

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

### Caption Generation Task
We evaluated the model's ability to generate textual captions describing music based on audio input. Key results include:

| Embedding Model               | Train  | Val   | Test  |
|-------------------------------|--------|-------|-------|
| Wav2Vec 2.0 (Frozen)          | 0.6158 | 0.6124| 0.6137|
| CLAP (Frozen)                 | 0.6318 | 0.6349| 0.6282|
| MERT (Frozen)                 | 0.6193 | 0.6306| 0.6149|
| Wav2Vec 2.0 (Unfrozen)        | 0.6121 | 0.6205| 0.6107|
| CLAP (Unfrozen)               | 0.5940 | 0.5895| 0.5892|
| MERT (Unfrozen)               | 0.5544 | 0.5600| 0.5536|

- **Key Observations**:
  - All embedding models perform poorly
  - The unfrozen embedding models perform worse than the frozen embedding models
  - The issue likely comes from issues with T5, not the embedding models themselves.

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
├── preprocessing/            # Extract wav files given YouTube ids (ytids) from metadata file
├── clap-to-t5/               # Caption generation using CLAP to T5 model
├── wav2vec-to-t5/            # Caption generation using Wav2Vec 2.0 to T5 model
├── mert-to-t5/               # Caption generation using MERT to T5 model
├── speecht5/                 # Caption generation attempt with SpeechT5 model (excluded from results)
├── plots/                    # Loss plots from training
├── dpo/                      # Progress towards using direct preference optimization on MusicGen
├── embedding_analysis/       # Training and evaluation scripts
├── music_samples/            # Sample wav files
├── plot_loss.py              # Plot the loss given loss files creating while training
├── docs/                     # Contains full final report
├── README.md                 # Project description and setup instructions
└── requirements.txt          # Required Python packages
```

---

## References

- Ding, Y. et al., "Audio Embedding Transfer Learning," 2023.
- Perera, M. et al., "Contrastive Learning for Audio Classification," 2024.
- GTZAN Dataset: [https://www.tensorflow.org/datasets/catalog/gtzan](https://www.tensorflow.org/datasets/catalog/gtzan)
- MusicCaps Dataset: [https://google-research.github.io/musiccaps/](https://google-research.github.io/musiccaps/)

For more details, see the full report in `docs/`.