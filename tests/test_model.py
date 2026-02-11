"""Tests for model architecture."""

import pytest
import torch

from contrastive_qa_verifier_with_adversarial_unanswerable.models.model import (
    ContrastiveQAVerifier
)


class TestContrastiveQAVerifier:
    """Tests for ContrastiveQAVerifier model."""

    def test_model_creation(self, sample_config):
        """Test model initialization."""
        model = ContrastiveQAVerifier(sample_config)

        assert model.embedding_dim == 384
        assert model.projection_dim == 256
        assert model.pooling_mode == "mean"
        assert model.temperature == 0.07

    def test_model_parameters(self, sample_config):
        """Test model has trainable parameters."""
        model = ContrastiveQAVerifier(sample_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params

    def test_pool_embeddings_mean(self, sample_config):
        """Test mean pooling."""
        model = ContrastiveQAVerifier(sample_config)

        batch_size = 2
        seq_len = 10
        hidden_dim = 384

        token_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        pooled = model.pool_embeddings(token_embeddings, attention_mask)

        assert pooled.shape == (batch_size, hidden_dim)

    def test_pool_embeddings_cls(self, sample_config):
        """Test CLS pooling."""
        config = sample_config.copy()
        config["model"]["pooling_mode"] = "cls"
        model = ContrastiveQAVerifier(config)

        batch_size = 2
        seq_len = 10
        hidden_dim = 384

        token_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        pooled = model.pool_embeddings(token_embeddings, attention_mask)

        assert pooled.shape == (batch_size, hidden_dim)
        # CLS pooling should return first token
        assert torch.allclose(pooled, token_embeddings[:, 0, :])

    def test_encode_question(self, sample_config, sample_batch):
        """Test question encoding."""
        model = ContrastiveQAVerifier(sample_config)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode_question(
                sample_batch["question_input_ids"],
                sample_batch["question_attention_mask"]
            )

        assert embeddings.shape == (2, 256)  # batch_size=2, projection_dim=256
        # Check normalized
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_answer(self, sample_config, sample_batch):
        """Test answer encoding."""
        model = ContrastiveQAVerifier(sample_config)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode_answer(
                sample_batch["answer_input_ids"],
                sample_batch["answer_attention_mask"]
            )

        assert embeddings.shape == (2, 256)
        # Check normalized
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward(self, sample_config, sample_batch):
        """Test forward pass."""
        model = ContrastiveQAVerifier(sample_config)
        model.eval()

        with torch.no_grad():
            question_embeds, answer_embeds = model(
                sample_batch["question_input_ids"],
                sample_batch["question_attention_mask"],
                sample_batch["answer_input_ids"],
                sample_batch["answer_attention_mask"]
            )

        assert question_embeds.shape == (2, 256)
        assert answer_embeds.shape == (2, 256)

    def test_compute_similarity(self, sample_config, sample_batch):
        """Test similarity computation."""
        model = ContrastiveQAVerifier(sample_config)
        model.eval()

        with torch.no_grad():
            question_embeds, answer_embeds = model(
                sample_batch["question_input_ids"],
                sample_batch["question_attention_mask"],
                sample_batch["answer_input_ids"],
                sample_batch["answer_attention_mask"]
            )

            similarity = model.compute_similarity(question_embeds, answer_embeds)

        assert similarity.shape == (2, 2)  # batch_size x batch_size

    def test_compute_loss(self, sample_config, sample_batch):
        """Test loss computation."""
        model = ContrastiveQAVerifier(sample_config)

        question_embeds, answer_embeds = model(
            sample_batch["question_input_ids"],
            sample_batch["question_attention_mask"],
            sample_batch["answer_input_ids"],
            sample_batch["answer_attention_mask"]
        )

        loss, metrics = model.compute_loss(
            question_embeds, answer_embeds, sample_batch["label"]
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
        assert "loss" in metrics
        assert metrics["num_positives"] == 2

    def test_predict(self, sample_config, sample_batch):
        """Test prediction."""
        model = ContrastiveQAVerifier(sample_config)
        model.eval()

        scores = model.predict(
            sample_batch["question_input_ids"],
            sample_batch["question_attention_mask"],
            sample_batch["answer_input_ids"],
            sample_batch["answer_attention_mask"]
        )

        assert scores.shape == (2,)
        assert torch.all(scores >= -1) and torch.all(scores <= 1)

    def test_model_gradients(self, sample_config, sample_batch):
        """Test that model computes gradients."""
        model = ContrastiveQAVerifier(sample_config)
        model.train()

        question_embeds, answer_embeds = model(
            sample_batch["question_input_ids"],
            sample_batch["question_attention_mask"],
            sample_batch["answer_input_ids"],
            sample_batch["answer_attention_mask"]
        )

        loss, _ = model.compute_loss(
            question_embeds, answer_embeds, sample_batch["label"]
        )

        loss.backward()

        # Check that gradients are computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients

    def test_model_device_transfer(self, sample_config):
        """Test moving model to device."""
        model = ContrastiveQAVerifier(sample_config)

        device = torch.device("cpu")
        model = model.to(device)

        # Check that parameters are on correct device
        for param in model.parameters():
            assert param.device == device
