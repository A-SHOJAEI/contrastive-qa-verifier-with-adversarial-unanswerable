"""Pytest configuration and fixtures."""

import pytest
import torch
from transformers import AutoTokenizer


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "question_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "answer_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "projection_dim": 256,
            "dropout": 0.1,
            "pooling_mode": "mean"
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "eval_steps": 50,
            "save_steps": 100,
            "logging_steps": 10,
            "seed": 42
        },
        "loss": {
            "temperature": 0.07,
            "margin": 0.5,
            "adversarial_weight": 0.3,
            "unanswerable_weight": 0.4
        },
        "data": {
            "max_question_length": 128,
            "max_answer_length": 256,
            "max_context_length": 512,
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "num_workers": 0,
            "cache_dir": "./data/cache",
            "adversarial_ratio": 0.3,
            "unanswerable_ratio": 0.2
        },
        "adversarial": {
            "num_negatives": 2,
            "entity_replacement": True,
            "semantic_similarity_threshold": 0.7,
            "max_edit_distance": 3,
            "adversarial_ratio": 0.3
        },
        "evaluation": {
            "metrics": ["f1", "precision", "recall", "accuracy", "auc"],
            "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
            "save_predictions": True
        },
        "paths": {
            "output_dir": "./test_models",
            "cache_dir": "./test_cache",
            "log_dir": "./test_logs",
            "results_dir": "./test_results"
        },
        "system": {
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
            "pin_memory": False
        }
    }


@pytest.fixture
def sample_data():
    """Sample QA data for testing."""
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When did World War II end?"
    ]

    answers = [
        "Paris",
        "William Shakespeare",
        "299,792,458 meters per second",
        "1945"
    ]

    contexts = [
        "Paris is the capital and most populous city of France.",
        "Romeo and Juliet is a tragedy written by William Shakespeare.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "World War II ended in 1945 with the surrender of Germany and Japan."
    ]

    labels = [1, 1, 1, 1]

    return questions, answers, contexts, labels


@pytest.fixture
def tokenizer():
    """Tokenizer fixture."""
    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def device():
    """Device fixture."""
    return torch.device("cpu")


@pytest.fixture
def sample_batch(tokenizer):
    """Sample batch for testing."""
    questions = ["What is AI?", "What is ML?"]
    answers = ["Artificial Intelligence", "Machine Learning"]

    question_encoded = tokenizer(
        questions,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    answer_encoded = tokenizer(
        answers,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "question_input_ids": question_encoded["input_ids"],
        "question_attention_mask": question_encoded["attention_mask"],
        "answer_input_ids": answer_encoded["input_ids"],
        "answer_attention_mask": answer_encoded["attention_mask"],
        "label": torch.tensor([1, 1], dtype=torch.long)
    }
