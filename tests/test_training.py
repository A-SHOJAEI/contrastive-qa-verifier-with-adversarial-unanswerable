"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader

from contrastive_qa_verifier_with_adversarial_unanswerable.models.model import (
    ContrastiveQAVerifier
)
from contrastive_qa_verifier_with_adversarial_unanswerable.training.trainer import (
    QAVerifierTrainer
)
from contrastive_qa_verifier_with_adversarial_unanswerable.data.loader import (
    QAVerificationDataset
)
from contrastive_qa_verifier_with_adversarial_unanswerable.evaluation.metrics import (
    QAVerifierEvaluator
)


class TestQAVerifierTrainer:
    """Tests for QAVerifierTrainer."""

    @pytest.fixture
    def trainer_setup(self, sample_config, sample_data, tokenizer):
        """Setup trainer with sample data."""
        questions, answers, contexts, labels = sample_data

        # Create datasets
        train_dataset = QAVerificationDataset(
            questions, answers, contexts, labels, tokenizer
        )
        val_dataset = QAVerificationDataset(
            questions[:2], answers[:2], contexts[:2], labels[:2], tokenizer
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        # Create model
        model = ContrastiveQAVerifier(sample_config)

        # Create trainer
        trainer = QAVerifierTrainer(
            model=model,
            config=sample_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=torch.device("cpu")
        )

        return trainer, train_loader, val_loader

    def test_trainer_creation(self, trainer_setup):
        """Test trainer initialization."""
        trainer, _, _ = trainer_setup

        assert trainer.num_epochs == 2
        assert trainer.learning_rate == 1e-4
        assert trainer.device.type == "cpu"
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_trainer_train_epoch(self, trainer_setup):
        """Test training for one epoch."""
        trainer, _, _ = trainer_setup

        initial_step = trainer.global_step
        metrics = trainer.train_epoch(epoch=0)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert trainer.global_step > initial_step

    def test_trainer_evaluate(self, trainer_setup):
        """Test evaluation."""
        trainer, _, _ = trainer_setup

        metrics = trainer.evaluate()

        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_trainer_save_checkpoint(self, trainer_setup, tmp_path):
        """Test checkpoint saving."""
        trainer, _, _ = trainer_setup
        trainer.output_dir = tmp_path

        checkpoint_name = "test_checkpoint"
        trainer.save_checkpoint(checkpoint_name)

        checkpoint_dir = tmp_path / checkpoint_name
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "model.pt").exists()
        assert (checkpoint_dir / "optimizer.pt").exists()
        assert (checkpoint_dir / "training_state.pt").exists()

    def test_trainer_load_checkpoint(self, trainer_setup, tmp_path):
        """Test checkpoint loading."""
        trainer, _, _ = trainer_setup
        trainer.output_dir = tmp_path

        # Save checkpoint
        checkpoint_name = "test_checkpoint"
        trainer.save_checkpoint(checkpoint_name)

        # Create new trainer and load
        new_model = ContrastiveQAVerifier(trainer.config)
        new_trainer = QAVerifierTrainer(
            model=new_model,
            config=trainer.config,
            train_dataloader=trainer.train_dataloader,
            val_dataloader=trainer.val_dataloader,
            device=torch.device("cpu")
        )

        checkpoint_path = tmp_path / checkpoint_name
        new_trainer.load_checkpoint(str(checkpoint_path))

        # Check that state was loaded
        assert new_trainer.global_step == trainer.global_step

    def test_optimizer_step(self, trainer_setup):
        """Test optimizer performs parameter updates."""
        trainer, train_loader, _ = trainer_setup

        # Get initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Perform one training step
        batch = next(iter(train_loader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}

        trainer.optimizer.zero_grad()

        question_embeds, answer_embeds = trainer.model(
            batch["question_input_ids"],
            batch["question_attention_mask"],
            batch["answer_input_ids"],
            batch["answer_attention_mask"]
        )

        loss, _ = trainer.model.compute_loss(
            question_embeds, answer_embeds, batch["label"]
        )

        loss.backward()
        trainer.optimizer.step()

        # Check that parameters changed
        params_changed = False
        for initial, current in zip(initial_params, trainer.model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break

        assert params_changed


class TestQAVerifierEvaluator:
    """Tests for QAVerifierEvaluator."""

    @pytest.fixture
    def evaluator_setup(self, sample_config, sample_data, tokenizer):
        """Setup evaluator with sample data."""
        questions, answers, contexts, labels = sample_data

        # Create dataset
        dataset = QAVerificationDataset(
            questions, answers, contexts, labels, tokenizer
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Create model
        model = ContrastiveQAVerifier(sample_config)

        # Create evaluator
        evaluator = QAVerifierEvaluator(
            model=model,
            config=sample_config,
            device=torch.device("cpu")
        )

        return evaluator, dataloader

    def test_evaluator_creation(self, evaluator_setup):
        """Test evaluator initialization."""
        evaluator, _ = evaluator_setup

        assert evaluator.device.type == "cpu"
        assert len(evaluator.thresholds) > 0

    def test_evaluator_predict(self, evaluator_setup):
        """Test prediction generation."""
        evaluator, dataloader = evaluator_setup

        scores, labels = evaluator.predict(dataloader)

        assert len(scores) == 4
        assert len(labels) == 4
        assert scores.shape == labels.shape

    def test_evaluator_compute_metrics(self, evaluator_setup):
        """Test metrics computation."""
        evaluator, _ = evaluator_setup

        # Sample predictions
        scores = torch.tensor([0.8, 0.6, 0.4, 0.2]).numpy()
        labels = torch.tensor([1, 1, 0, 0]).numpy()

        metrics = evaluator.compute_metrics(scores, labels, threshold=0.5)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_evaluator_evaluate(self, evaluator_setup):
        """Test full evaluation."""
        evaluator, dataloader = evaluator_setup

        results = evaluator.evaluate(dataloader, return_predictions=True)

        assert "scores" in results
        assert "labels" in results
        assert "metrics_by_threshold" in results
        assert "best_threshold" in results
        assert "best_metrics" in results
        assert "num_samples" in results

        assert results["num_samples"] == 4
        assert len(results["metrics_by_threshold"]) > 0

    def test_evaluator_answerable_unanswerable_metrics(self, evaluator_setup):
        """Test answerable/unanswerable specific metrics."""
        evaluator, _ = evaluator_setup

        scores = torch.tensor([0.8, 0.6, 0.4, 0.2]).numpy()
        labels = torch.tensor([1, 1, 0, 0]).numpy()

        metrics = evaluator.compute_answerable_unanswerable_metrics(
            scores, labels, threshold=0.5
        )

        assert "answerable_accuracy" in metrics
        assert "unanswerable_accuracy" in metrics
        assert "num_answerable" in metrics
        assert "num_unanswerable" in metrics

        assert metrics["num_answerable"] == 2
        assert metrics["num_unanswerable"] == 2

    def test_evaluator_save_predictions(self, evaluator_setup, tmp_path):
        """Test saving predictions."""
        evaluator, _ = evaluator_setup

        scores = torch.tensor([0.8, 0.6, 0.4, 0.2]).numpy()
        labels = torch.tensor([1, 1, 0, 0]).numpy()

        output_path = tmp_path / "predictions.json"
        evaluator.save_predictions(scores, labels, str(output_path))

        assert output_path.exists()

        # Load and verify
        import json
        with open(output_path) as f:
            data = json.load(f)

        assert "scores" in data
        assert "labels" in data
        assert len(data["scores"]) == 4
        assert len(data["labels"]) == 4
