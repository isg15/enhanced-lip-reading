# Enhanced Lip Reading with Sentiment Analysis

A multimodal language modeling approach for lip reading enhanced with sentiment analysis in motion-informed context.

## Project Overview

This project implements an advanced lip-reading system that combines:
- 3D ResNet-50 based visual feature extraction
- Multi-motion-informed context generation
- Sentiment analysis for emotion-aware transcription
- Transformer-based decoder for character-level prediction

## Objectives

- Enhance lip reading precision by mitigating lip pattern ambiguities
- Achieve deeper emotional and contextual understanding in lip reading
- Enrich lip reading with multimodal data capturing non-verbal communication
- Advance visual analysis capabilities with spatiotemporal feature recognition
- Improve language generation quality through external models

## Architecture

### Main Components

1. **Video Preprocessing Module**
   - Face detection and ROI extraction using RetinaFace and FANPredictor
   - Frame sampling and normalization
   - TFRecords generation for efficient data loading

2. **Visual Module (3D ResNet-50)**
   - Local-Pool Attention (LPA) blocks for temporal feature extraction
   - Weighted Dynamic Aggregation (WDA) for multi-scale fusion
   - Post Feed-Forward Network (Post-FFN) for context refinement

3. **Decoder (Transformer-based)**
   - Character-level language model with positional encoding
   - 6-layer transformer decoder with multi-head attention
   - Cross-attention mechanism for visual-linguistic integration

4. **Sentiment Analysis Module**
   - CNN-based emotion recognition on RAVDESS dataset
   - 7 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprise
   - Mean face subtraction for improved feature extraction

5. **Inference Module**
   - Beam search decoding with configurable width
   - Combined probability generation from visual and sentiment features
   - External language model integration for fluency

## Datasets

- **LRW (Lip Reading in the Wild)**: 450,000 word-level utterances from BBC recordings
- **LRS2-BBC**: 143,000 utterances with 41,000-word vocabulary
- **RAVDESS**: 7,356 recordings with 8 emotion categories for sentiment analysis

## Performance Metrics

| Metric | Score |
|--------|-------|
| Word Error Rate (WER) | 45.9% |
| Character Error Rate (CER) | 63.7% |
| Sentiment Analysis Accuracy | 89% |

### Detailed Sentiment Analysis Results

**With Mean Face Subtraction:**
- Overall Accuracy: 89%
- Best performing emotions: Happy (96% F1), Disgust (92% F1), Fear (88% F1)

**Without Mean Face Subtraction:**
- Overall Accuracy: 82%

## Technologies

**Deep Learning Frameworks:**
- TensorFlow 2.x
- PyTorch 1.12+

**Computer Vision:**
- OpenCV
- RetinaFace (face detection)
- FANPredictor (facial landmark detection)

**Data Processing:**
- NumPy
- Pandas
- TFRecords

**Visualization:**
- Matplotlib

## Installation

### Prerequisites
```bash
Python >= 3.8
CUDA >= 11.0 (for GPU support)
```

### Setup
```bash
# Clone the repository
git clone https://github.com/isg15/enhanced-lip-reading.git
cd enhanced-lip-reading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
enhanced-lip-reading/
├── preprocessing/
│   ├── video_preprocessing.py       # Video frame extraction and ROI detection
│   ├── landmark_detection.py        # Facial landmark detection
│   ├── roi_extraction.py            # Mouth region extraction
│   └── data_preparation.py          # TFRecords generation
├── models/
│   ├── resnet.py                    # 3D ResNet-50 implementation
│   ├── lpa_block.py                 # Local-Pool Attention blocks
│   ├── visual_module.py             # Complete visual encoder
│   ├── decoder.py                   # Transformer decoder
│   ├── sentiment_analysis.py        # CNN-based emotion classifier
│   └── combined_model.py            # Integrated lip-reading model
├── training/
│   ├── pretrain_decoder.py          # Language model pretraining
│   ├── pretrain_visual.py           # Visual module pretraining
│   ├── train_sentiment.py           # Sentiment analysis training
│   └── train_integration.py         # End-to-end model training
├── inference/
│   ├── beam_search.py               # Beam search decoder
│   └── predict.py                   # Inference pipeline
├── utils/
│   ├── metrics.py                   # WER, CER calculation
│   ├── data_loader.py               # Custom data loaders
│   └── visualization.py             # Result visualization
├── notebooks/
│   └── Lip_Reading.ipynb            # Main development notebook
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Usage

### 1. Data Preprocessing
```bash
# Extract and preprocess LRS2 dataset
python preprocessing/video_preprocessing.py \
    --data-dir /path/to/lrs2 \
    --output-dir /path/to/preprocessed \
    --dataset lrs2 \
    --subset train

# Process RAVDESS for sentiment analysis
python preprocessing/data_preparation.py \
    --data-dir /path/to/ravdess \
    --output-dir /path/to/ravdess_processed
```

### 2. Model Training

**Step 1: Pretrain Decoder (Language Model)**
```bash
python training/pretrain_decoder.py \
    --corpus-path /path/to/text_corpus \
    --output-dir ./checkpoints/decoder \
    --epochs 10 \
    --batch-size 32
```

**Step 2: Pretrain Visual Module**
```bash
python training/pretrain_visual.py \
    --data-dir /path/to/preprocessed \
    --output-dir ./checkpoints/visual \
    --epochs 10 \
    --batch-size 16
```

**Step 3: Train Sentiment Analysis**
```bash
python training/train_sentiment.py \
    --data-dir /path/to/ravdess_processed \
    --output-dir ./checkpoints/sentiment \
    --epochs 10 \
    --use-mean-face
```

**Step 4: Integrate and Fine-tune**
```bash
python training/train_integration.py \
    --decoder-checkpoint ./checkpoints/decoder/model.pth \
    --visual-checkpoint ./checkpoints/visual/model.pth \
    --sentiment-checkpoint ./checkpoints/sentiment/model.h5 \
    --data-dir /path/to/preprocessed \
    --output-dir ./checkpoints/integrated \
    --epochs 20
```

### 3. Inference
```bash
# Single video inference
python inference/predict.py \
    --video-path /path/to/video.mp4 \
    --model-checkpoint ./checkpoints/integrated/best_model.pth \
    --beam-width 12 \
    --output transcript.txt

# Batch inference
python inference/predict.py \
    --video-dir /path/to/videos \
    --model-checkpoint ./checkpoints/integrated/best_model.pth \
    --beam-width 12 \
    --output-dir ./results
```

## Model Architecture Details

### Visual Module Components

**3D ResNet-50 Backbone:**
- 5 residual blocks with varying channel dimensions (64, 128, 256, 512, 1024)
- 3D convolutions for spatiotemporal feature extraction
- Batch normalization and ReLU activations

**Local-Pool Attention (LPA):**
```
Input: [batch, time, channels, height, width]
↓
Max Pooling (spatial dimension reduction)
↓
Projection to decoder dimension
↓
Positional Encoding
↓
Self-Attention (temporal dependencies)
↓
Output: [batch, time, decoder_dim]
```

**Weighted Dynamic Aggregation (WDA):**
- Learnable weights for combining outputs from multiple LPA blocks
- Adaptive fusion of multi-scale temporal features

### Decoder Architecture

**Transformer Decoder:**
- 6 layers with 16 attention heads
- Model dimension: 1024
- Feed-forward dimension: 4096
- Dropout: 0.1

**Character-Level Generation:**
- Vocabulary size: 40 characters (A-Z, space, apostrophe, special tokens)
- Start token: [SOS] (ID: 39)
- Padding token: [PAD] (ID: 0)
- Unknown token: [UNK] (ID: 1)

### Sentiment Analysis Network

**CNN Architecture:**
```
Conv2D(32, 5×5) → BatchNorm → ReLU → MaxPool
↓
Conv2D(64, 5×5) → BatchNorm → ReLU → MaxPool
↓
Dropout(0.5)
↓
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool
↓
Dropout(0.5)
↓
Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool
↓
Dropout(0.5)
↓
Flatten → Dense(128) → Dense(7) → Softmax
```

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate (Decoder) | 0.001 |
| Learning Rate (Visual) | 0.0001 |
| Batch Size | 16-32 |
| Epochs (Pretraining) | 10 |
| Epochs (Integration) | 20 |
| Loss Function | Cross-Entropy |
| Beam Width (Inference) | 12 |

### Data Augmentation

**Video Processing:**
- Horizontal flip (probability: 0.5)
- Random crop around detected landmarks
- Temporal masking (Adaptive Time Mask)

**Sentiment Analysis:**
- Random rotation (±10 degrees)
- Random brightness adjustment
- Mean face subtraction

## Evaluation Metrics

### Word Error Rate (WER)
```
WER = (Substitutions + Deletions + Insertions) / Total Words
```

### Character Error Rate (CER)
```
CER = (Substitutions + Deletions + Insertions) / Total Characters
```

### Sentiment Analysis Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

## Comparative Analysis

| Model | WER Score | Dataset |
|-------|-----------|---------|
| WAS (Pretrained) | 70.4% | LRS2 |
| TM-seq2seq (Pretrained) | 49.8% | LRS2 |
| TM-seq2seq+LM | 56.1% | LRS2 |
| TM-ResNet | 47.6% | LRS2 |
| TCN | 51.1% | LRS2 |
| TM-ResNet+Subword | 46.4% | LRS2 |
| **Our Model** | **45.9%** | LRS2 |

## Key Features

### Piece-Wise Pre-Training Strategy

1. **Decoder Pre-training**: Train as language model on text corpus
2. **Visual Module Pre-training**: Train as context generator on video clips
3. **Gap Bridging**: Connect through shared softmax layer
4. **Fine-tuning**: End-to-end training with cross-attention

### Multi-Motion-Informed Context

- Captures lip movements at multiple temporal scales
- Aggregates features from 5 ResNet blocks
- Applies Local-Pool Attention at each scale
- Dynamically weights contributions via WDA

### Sentiment-Aware Transcription

- Facial emotion analysis guides text generation
- Combines visual lip-reading with emotion scores
- Improves context understanding in ambiguous scenarios


## Future Work

- Expand character vocabulary for improved out-of-vocabulary word handling
- Implement advanced 3D visual backends (e.g., I3D, SlowFast)
- Explore temporal attention mechanisms for better frame-to-character alignment
- Optimize for real-time inference on edge devices
- Extend to multi-language lip reading

