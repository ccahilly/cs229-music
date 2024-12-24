Investigating Audio Encoding Models' Effectiveness on Downstream Tasks

Authors

Caroline Cahilly, Caleb Liu, Thomas YimDepartment of Computer Science, Stanford UniversityContact: ccahilly@stanford.edu, calebliu@stanford.edu, yimt@stanford.edu

Introduction

Music plays an integral role in human experiences. From sparking memories to enhancing performance, music's impact is undeniable. Machine learning (ML)-based systems, such as music recommendation engines, rely on robust audio embeddings to represent music for downstream tasks effectively.

We explore the effectiveness of various audio embedding frameworks—MERT, Wav2Vec 2.0, and CLAP—on two tasks: genre classification and music captioning. These tasks evaluate the models' abilities to extract musically relevant information and generate semantically meaningful audio representations.

Related Work

Prior research highlights the benefits of music-specific embeddings for classification tasks. Key references include:

Ding et al. (2023): Studied audio embeddings' role in knowledge transfer, achieving up to 15% performance improvement.

Perera et al. (2024): Demonstrated the advantages of contrastively learned embeddings over traditional hand-crafted features.

Our work builds on this foundation by evaluating transformer-based embeddings to capture richer musical properties.

Datasets

Genre Classification

Dataset: GTZAN dataset

Details: 1000 audio clips (30 seconds each) across 10 genres.

Split: 800 training, 200 test examples.

Music Captioning

Dataset: MusicCaps

Details: 5360 examples of 10-second clips paired with descriptive captions.

Split: 70-10-20 (train-validation-test).

Processing: WAV format, resampling rates based on embedding models (CLAP: 48kHz, Wav2Vec 2.0: 24kHz, MERT: 24kHz). Mono audio normalization applied.

Methods

Audio Embedding Models

CLAP (Contrastive Language-Audio Pretraining): Learns embeddings via contrastive loss to align audio-text pairs.

Wav2Vec 2.0: Self-supervised model that learns from raw speech signals.

MERT (Acoustic Music Understanding Model): Leverages both acoustic and pitch-based features through transformer-based architecture.

Tasks

Genre Classification

Embedding dimensions are projected, normalized, and input to an MLP classifier.

Loss Function: Cross-entropy loss.

Optimizer: Adam.

Text Generation

Embeddings are passed to T5, a transformer-based model.

Loss Function: Cross-entropy loss.

Optimizer: Adam.

Experiments and Results

Genre Classification

Setup

Six experiments with combinations of frozen/unfrozen embeddings for each model.

Epochs: 10.

Learning Rates: 1e-5 (optimal).

Results

Model

Train Accuracy (%)

Test Accuracy (%)

Wav2Vec 2.0 (Frozen)

44.25

43.50

CLAP (Frozen)

37.00

36.63

MERT (Frozen)

72.12

66.50

Wav2Vec 2.0 (Unfrozen)

92.75

67.50

CLAP (Unfrozen)

63.75

43.00

MERT (Unfrozen)

100.00

84.00

MERT outperformed other models, especially when fine-tuned. Pretrained embeddings on music-specific data (MERT) yielded better results than speech (Wav2Vec 2.0) or general audio (CLAP).

Embedding Visualization

Principal Component Analysis (PCA) visualizations highlight MERT's ability to cluster genres more distinctly after fine-tuning.

Discussion

Embedding Models: MERT's music-specific pretraining offered superior performance.

Fine-tuning: Unfreezing encoder weights significantly improved accuracy across models.

Challenges: CLAP struggled due to high embedding similarity across samples.

Future Work

Explore hybrid embeddings combining temporal and spectral features.

Test embedding models on additional downstream tasks like mood detection and instrument recognition.

Integrate richer evaluation metrics beyond classification accuracy.

References

Ding, 2023. "Knowledge Transfer in Music Classification." Paper link.

Perera, 2024. "Contrastive Audio Embeddings." Paper link.

GTZAN Dataset. Link.

MusicCaps Dataset. Link.

Repository Structure

.
├── data
│   ├── gtzan
│   ├── musiccaps
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
├── models
│   ├── CLAP
│   ├── Wav2Vec2
│   ├── MERT
├── results
│   ├── plots
│   ├── logs
└── README.md

Usage

Setup

git clone https://github.com/ccahilly/cs229-music.git
cd cs229-music
pip install -r requirements.txt

Running Experiments

python src/train.py --model mert --task genre_classification

Evaluating Results

python src/evaluate.py --model mert --task genre_classification

License

This project is licensed under the MIT License. See LICENSE for details.

Acknowledgments

We thank Stanford University and the CS229 teaching staff for their support.
