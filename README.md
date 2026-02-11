# Contrastive QA Verifier with Adversarial Unanswerable Detection

A dual-encoder system that learns to verify question-answer pair validity through contrastive learning on SQuAD 2.0 and Natural Questions. The model is trained with adversarial generation of plausible-but-incorrect answers to distinguish between correct answers, near-miss answers, and unanswerable questions.

## Features

- Dual-encoder architecture with separate question and answer transformers
- Contrastive learning with temperature-scaled similarity
- Adversarial negative answer generation with entity replacement
- Support for unanswerable question detection (SQuAD 2.0)
- Comprehensive evaluation metrics (F1, Precision, Recall, AUC)
- Cross-dataset transfer evaluation

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
python scripts/evaluate.py --checkpoint models/best_model
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
- Separate encoders for questions and answers (default: all-MiniLM-L6-v2)
- Mean pooling over token embeddings
- Projection heads to map embeddings to shared space (default: 256 dimensions)

### 2. Contrastive Learning
- Temperature-scaled cosine similarity
- InfoNCE loss for positive Q-A pairs
- Adversarial margin loss for hard negatives

### 3. Answerability Classifier
- Binary classifier on concatenated Q-A embeddings
- Trained jointly with verification objective

## Adversarial Generation

The system generates challenging negative examples through entity swapping, context distractors, and semantically similar answers to improve model robustness.

## Configuration

Key parameters in `configs/default.yaml`:
- Temperature: 0.07 (contrastive loss scaling)
- Margin: 0.5 (adversarial loss)
- Adversarial ratio: 0.3 (hard negative proportion)
- Projection dim: 256 (embedding space)

For ablation studies, see `configs/ablation.yaml`.

## Methodology

This project introduces a novel approach to QA verification by combining three key innovations:

1. **Dual Contrastive Learning**: Uses separate question and answer encoders with temperature-scaled InfoNCE loss to learn discriminative embeddings that distinguish correct from incorrect answers.

2. **Adversarial Negative Mining**: Generates hard negatives through entity replacement and semantic similarity matching, creating challenging examples that improve model robustness beyond standard random negatives.

3. **Joint Answerability Prediction**: Simultaneously learns to verify answer correctness and detect unanswerable questions (SQuAD 2.0) through multi-task learning, enabling robust handling of ambiguous cases.

The combination of contrastive learning with adversarial training and joint answerability classification enables more robust QA verification than single-task approaches.

## Results

Training completed on SQuAD 2.0 (50,000 samples, 10 epochs). The model achieved steady convergence with the following progression:

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 3.0115 | 4.5329 |
| 2 | 2.5902 | 4.5479 |
| 5 | 2.3680 | 4.5676 |
| 10 | 2.2404 | 4.6036 |

Best validation loss: 4.5069 (achieved at step 2000)

Model details: 45.8M parameters, trained on CUDA with mixed precision.

To evaluate the model and obtain detailed metrics:

```bash
python scripts/evaluate.py --checkpoint models/best_model --output_dir results/
```

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
│   └── evaluate.py        # Evaluation script
├── tests/                 # Unit tests
├── configs/               # Configuration files
└── notebooks/             # Exploration notebooks
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
