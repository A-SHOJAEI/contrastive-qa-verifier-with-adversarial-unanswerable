"""Training utilities for QA verifier."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class QAVerifierTrainer:
    """Trainer for QA verification model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer.

        Args:
            model: QA verifier model.
            config: Configuration dictionary.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            device: Device to train on (cuda/cpu).
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Device setup
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model.to(self.device)

        # Training config
        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_epochs", 10)
        self.learning_rate = train_config.get("learning_rate", 1e-4)
        self.weight_decay = train_config.get("weight_decay", 0.01)
        self.warmup_steps = train_config.get("warmup_steps", 500)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        self.eval_steps = train_config.get("eval_steps", 500)
        self.save_steps = train_config.get("save_steps", 1000)
        self.logging_steps = train_config.get("logging_steps", 100)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # Mixed precision training
        self.use_amp = config.get("system", {}).get("mixed_precision", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Output directory
        self.output_dir = Path(config.get("paths", {}).get("output_dir", "./models"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MLflow tracking
        self.use_mlflow = False
        try:
            import mlflow
            mlflow_config = config.get("mlflow", {})
            if mlflow_config:
                mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "./mlruns"))
                mlflow.set_experiment(mlflow_config.get("experiment_name", "contrastive-qa-verifier"))
                self.use_mlflow = True
                logger.info("MLflow tracking enabled")
        except ImportError:
            logger.warning("MLflow not available, tracking disabled")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")

        logger.info(f"Initialized trainer on device: {self.device}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    question_embeds, answer_embeds = self.model(
                        batch["question_input_ids"],
                        batch["question_attention_mask"],
                        batch["answer_input_ids"],
                        batch["answer_attention_mask"]
                    )
                    loss, metrics = self.model.compute_loss(
                        question_embeds, answer_embeds, batch["label"]
                    )
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                question_embeds, answer_embeds = self.model(
                    batch["question_input_ids"],
                    batch["question_attention_mask"],
                    batch["answer_input_ids"],
                    batch["answer_attention_mask"]
                )
                loss, metrics = self.model.compute_loss(
                    question_embeds, answer_embeds, batch["label"]
                )
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": total_loss / num_batches})

            # Logging
            if self.global_step % self.logging_steps == 0 and self.use_mlflow:
                try:
                    import mlflow
                    mlflow.log_metric("train_loss", total_loss / num_batches, step=self.global_step)
                except Exception as e:
                    logger.debug(f"MLflow logging failed: {e}")

            # Evaluation
            if self.global_step % self.eval_steps == 0:
                val_metrics = self.evaluate()
                logger.info(f"Step {self.global_step} - Val Loss: {val_metrics['loss']:.4f}")

                if self.use_mlflow:
                    try:
                        import mlflow
                        for key, value in val_metrics.items():
                            mlflow.log_metric(f"val_{key}", value, step=self.global_step)
                    except Exception as e:
                        logger.debug(f"MLflow logging failed: {e}")

                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best_model")

                self.model.train()

            # Checkpointing
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")

        return {"loss": total_loss / num_batches}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                question_embeds, answer_embeds = self.model(
                    batch["question_input_ids"],
                    batch["question_attention_mask"],
                    batch["answer_input_ids"],
                    batch["answer_attention_mask"]
                )

                loss, metrics = self.model.compute_loss(
                    question_embeds, answer_embeds, batch["label"]
                )

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches}

    def train(self) -> None:
        """Run complete training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.start_run()
                mlflow.log_params({
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.config.get("training", {}).get("batch_size", 32),
                    "model_type": "ContrastiveQAVerifier"
                })
            except Exception as e:
                logger.warning(f"MLflow run start failed: {e}")

        try:
            for epoch in range(self.num_epochs):
                train_metrics = self.train_epoch(epoch)
                logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")

                # End of epoch evaluation
                val_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}")

                if self.use_mlflow:
                    try:
                        import mlflow
                        mlflow.log_metric("epoch_train_loss", train_metrics["loss"], step=epoch)
                        mlflow.log_metric("epoch_val_loss", val_metrics["loss"], step=epoch)
                    except Exception as e:
                        logger.debug(f"MLflow logging failed: {e}")

            # Save final model
            self.save_checkpoint("final_model")

        finally:
            if self.use_mlflow:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception as e:
                    logger.debug(f"MLflow run end failed: {e}")

        logger.info("Training completed")

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Args:
            name: Checkpoint name.
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save training state
        state_path = checkpoint_dir / "training_state.pt"
        torch.save({
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, state_path)

        logger.info(f"Saved checkpoint: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        checkpoint_dir = Path(checkpoint_path)

        # Load model
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")

        # Load optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info(f"Loaded optimizer from {optimizer_path}")

        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.best_val_loss = state["best_val_loss"]
            logger.info(f"Loaded training state from {state_path}")
