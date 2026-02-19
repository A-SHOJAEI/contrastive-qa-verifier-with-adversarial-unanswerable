# Contrastive QA Verifier with Adversarial Unanswerable Detection

A dual-encoder system that learns to verify question-answer pair validity through contrastive learning on SQuAD 2.0. The model is trained with adversarial generation of plausible-but-incorrect answers to distinguish between correct answers, near-miss answers, and unanswerable questions.

## Features

- Dual-encoder architecture with separate question and answer transformers
- Contrastive learning with temperature-scaled similarity (InfoNCE loss)
- Adversarial negative answer generation with entity replacement and answer swapping
- Support for unanswerable question detection (SQuAD 2.0)
- Comprehensive evaluation metrics (F1, Precision, Recall, AUC)
- MLflow experiment tracking

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train.py
```

The training script will:
- Load and preprocess SQuAD 2.0 dataset
- Generate adversarial negative examples
- Train the dual-encoder model with contrastive loss
- Save checkpoints to `models/`
- Log metrics to MLflow

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/best_model --output_dir results/
```

### Inference

```bash
python scripts/predict.py --checkpoint models/best_model \
  --question "What is the capital of France?" \
  --answer "Paris"
```

## Model Architecture

The system consists of three main components:

### 1. Dual Encoder
- Separate encoders for questions and answers (all-MiniLM-L6-v2)
- Mean pooling over token embeddings
- Projection heads mapping to a shared 256-dimensional space
- Total parameters: 45.8M

### 2. Contrastive Learning
- Temperature-scaled cosine similarity (tau = 0.07)
- InfoNCE loss for positive Q-A pairs
- Adversarial margin loss for hard negatives (margin = 0.5)

### 3. Answerability Classifier
- Binary classifier on concatenated Q-A embeddings
- Trained jointly with verification objective (weight = 0.4)

## Adversarial Generation

The system generates challenging negative examples through entity swapping, context distractors, and semantically similar answers to improve model robustness. During training, 10,537 adversarial examples and 7,048 answer-swapped examples were generated from the base SQuAD 2.0 training split.

## Configuration

Key parameters in `configs/default.yaml`:
- Temperature: 0.07 (contrastive loss scaling)
- Margin: 0.5 (adversarial loss)
- Adversarial ratio: 0.3 (hard negative proportion)
- Projection dim: 256 (embedding space)

For ablation studies, see `configs/ablation.yaml`.

## Methodology

This project introduces an approach to QA verification by combining three techniques:

1. **Dual Contrastive Learning**: Uses separate question and answer encoders with temperature-scaled InfoNCE loss to learn discriminative embeddings that distinguish correct from incorrect answers.

2. **Adversarial Negative Mining**: Generates hard negatives through entity replacement and answer swapping, creating challenging examples that push the model beyond trivial discrimination.

3. **Joint Answerability Prediction**: Simultaneously learns to verify answer correctness and detect unanswerable questions through multi-task learning on SQuAD 2.0.

## Results

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Encoder | all-MiniLM-L6-v2 |
| Model Parameters | 45,755,136 |
| Dataset | SQuAD 2.0 |
| Training Samples | 67,585 (incl. adversarial) |
| Validation Samples | 5,000 |
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Device | CUDA |

### Training Loss Progression

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 3.0181    | 4.5574   |
| 2     | 2.5884    | 4.5543   |
| 3     | 2.4790    | 4.5357   |
| 4     | 2.4157    | 4.5981   |
| 5     | 2.3681    | 4.5537   |
| 6     | 2.3323    | 4.5908   |
| 7     | 2.3084    | 4.6251   |
| 8     | 2.2791    | 4.6186   |
| 9     | 2.2570    | 4.5828   |
| 10    | 2.2378    | 4.6025   |

Best validation loss: **4.5209** at step 1,500 (Epoch 1).

The training loss decreased steadily from 3.02 to 2.24, indicating the model learned meaningful representations on the training data. Validation loss plateaued early around 4.52--4.60, with the best checkpoint occurring at step 1,500.

### Evaluation Results (Validation Set, 5,000 samples)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.5064 |
| AUC       | 0.5035 |
| Precision | 0.0000 |
| Recall    | 0.0000 |
| F1        | 0.0000 |

The model's similarity scores cluster below the tested classification thresholds (0.3--0.7), resulting in all-negative predictions. The AUC of 0.5035 indicates limited discriminative power on the validation set at this stage of development.

### Analysis

The gap between training loss convergence and validation performance suggests the model overfits to training-specific patterns without generalizing the contrastive similarity structure to held-out data. Potential directions for improvement include:

- Lower temperature values for sharper contrastive distributions
- Larger projection dimensions or alternative pooling strategies
- Hard negative mining with dynamic difficulty scheduling
- Pre-training the encoders jointly before adding the contrastive objective
- Extended hyperparameter search over learning rate and warmup schedules

## Project Structure

```
contrastive-qa-verifier-with-adversarial-unanswerable/
├── src/contrastive_qa_verifier_with_adversarial_unanswerable/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture
│   ├── training/          # Training loop and trainer
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Configuration and utilities
├── scripts/
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── predict.py         # Inference script
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── results/               # Evaluation results
└── notebooks/             # Exploration notebooks
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
