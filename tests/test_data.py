"""Tests for data loading and preprocessing."""

import pytest
import torch

from contrastive_qa_verifier_with_adversarial_unanswerable.data.loader import (
    QAVerificationDataset,
    QADataLoader
)
from contrastive_qa_verifier_with_adversarial_unanswerable.data.preprocessing import (
    QAPreprocessor,
    AdversarialGenerator
)


class TestQAVerificationDataset:
    """Tests for QAVerificationDataset."""

    def test_dataset_creation(self, sample_data, tokenizer):
        """Test dataset creation."""
        questions, answers, contexts, labels = sample_data

        dataset = QAVerificationDataset(
            questions, answers, contexts, labels, tokenizer, max_length=128
        )

        assert len(dataset) == len(questions)
        assert dataset.questions == questions
        assert dataset.answers == answers

    def test_dataset_getitem(self, sample_data, tokenizer):
        """Test dataset item retrieval."""
        questions, answers, contexts, labels = sample_data

        dataset = QAVerificationDataset(
            questions, answers, contexts, labels, tokenizer, max_length=128
        )

        item = dataset[0]

        assert "question_input_ids" in item
        assert "question_attention_mask" in item
        assert "answer_input_ids" in item
        assert "answer_attention_mask" in item
        assert "label" in item

        assert item["question_input_ids"].shape[0] == 128
        assert item["answer_input_ids"].shape[0] == 128
        assert isinstance(item["label"], torch.Tensor)

    def test_dataset_length_mismatch(self, tokenizer):
        """Test dataset with mismatched lengths."""
        questions = ["Q1", "Q2"]
        answers = ["A1"]
        contexts = ["C1", "C2"]
        labels = [1, 0]

        with pytest.raises(ValueError):
            QAVerificationDataset(
                questions, answers, contexts, labels, tokenizer
            )


class TestQADataLoader:
    """Tests for QADataLoader."""

    def test_dataloader_creation(self, sample_config, tokenizer):
        """Test dataloader creation."""
        loader = QADataLoader(sample_config, tokenizer)

        assert loader.config == sample_config
        assert loader.tokenizer == tokenizer

    def test_create_dataloader(self, sample_config, sample_data, tokenizer):
        """Test creating PyTorch DataLoader."""
        questions, answers, contexts, labels = sample_data

        dataset = QAVerificationDataset(
            questions, answers, contexts, labels, tokenizer
        )

        loader = QADataLoader(sample_config, tokenizer)
        dataloader = loader.create_dataloader(dataset, batch_size=2, shuffle=False)

        assert len(dataloader) == 2  # 4 samples / batch_size 2

        # Test iteration
        batch = next(iter(dataloader))
        assert batch["question_input_ids"].shape[0] == 2
        assert batch["label"].shape[0] == 2

    def test_prepare_dataloaders(self, sample_config, sample_data, tokenizer):
        """Test preparing multiple dataloaders."""
        loader = QADataLoader(sample_config, tokenizer)

        dataloaders = loader.prepare_dataloaders(
            sample_data, sample_data, sample_data
        )

        assert "train" in dataloaders
        assert "val" in dataloaders
        assert "test" in dataloaders


class TestQAPreprocessor:
    """Tests for QAPreprocessor."""

    def test_preprocessor_creation(self, sample_config):
        """Test preprocessor creation."""
        preprocessor = QAPreprocessor(sample_config)

        assert preprocessor.config == sample_config
        assert preprocessor.max_question_length == 128
        assert preprocessor.max_answer_length == 256

    def test_split_data(self, sample_config, sample_data):
        """Test data splitting."""
        preprocessor = QAPreprocessor(sample_config)
        questions, answers, contexts, labels = sample_data

        train_data, val_data, test_data = preprocessor.split_data(
            questions, answers, contexts, labels
        )

        # Check that all data is accounted for
        total_samples = len(train_data[0]) + len(val_data[0]) + len(test_data[0])
        assert total_samples == len(questions)

        # Check that train is largest
        assert len(train_data[0]) >= len(val_data[0])
        assert len(train_data[0]) >= len(test_data[0])


class TestAdversarialGenerator:
    """Tests for AdversarialGenerator."""

    def test_generator_creation(self, sample_config):
        """Test adversarial generator creation."""
        generator = AdversarialGenerator(sample_config)

        assert generator.config == sample_config
        assert generator.num_negatives == 2

    def test_generate_negative_answer(self, sample_config):
        """Test negative answer generation."""
        generator = AdversarialGenerator(sample_config)

        answer = "Paris"
        context = "Paris is the capital of France. London is the capital of UK."
        question = "What is the capital of France?"

        negative = generator.generate_negative_answer(answer, context, question)

        assert negative != answer
        assert len(negative) > 0

    def test_generate_adversarial_examples(self, sample_config, sample_data):
        """Test adversarial example generation."""
        generator = AdversarialGenerator(sample_config)
        questions, answers, contexts, labels = sample_data

        aug_questions, aug_answers, aug_contexts, aug_labels = generator.generate_adversarial_examples(
            questions, answers, contexts, labels
        )

        # Should have more samples after augmentation
        assert len(aug_questions) >= len(questions)
        assert len(aug_answers) == len(aug_questions)
        assert len(aug_contexts) == len(aug_questions)
        assert len(aug_labels) == len(aug_questions)

    def test_swap_answers(self, sample_config, sample_data):
        """Test answer swapping."""
        generator = AdversarialGenerator(sample_config)
        questions, answers, contexts, labels = sample_data

        aug_questions, aug_answers, aug_contexts, aug_labels = generator.swap_answers(
            questions, answers, contexts, labels
        )

        # Should have more samples after swapping
        assert len(aug_questions) >= len(questions)
        assert len(aug_answers) == len(aug_questions)
