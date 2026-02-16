"""Evaluation metrics for QA verifier."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QAVerifierEvaluator:
    """Evaluator for QA verification model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize evaluator.

        Args:
            model: QA verifier model.
            config: Configuration dictionary.
            device: Device to run evaluation on.
        """
        self.model = model
        self.config = config
        self.eval_config = config.get("evaluation", {})

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # Evaluation thresholds
        self.thresholds = self.eval_config.get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])

        logger.info(f"Initialized evaluator on device: {self.device}")

    def predict(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions on a dataset.

        Args:
            dataloader: DataLoader for evaluation.

        Returns:
            Tuple of (predictions, labels).
        """
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch_dict = {k: v.to(self.device) for k, v in batch.items()}

                # Get prediction scores
                scores = self.model.predict(
                    batch_dict["question_input_ids"],
                    batch_dict["question_attention_mask"],
                    batch_dict["answer_input_ids"],
                    batch_dict["answer_attention_mask"]
                )

                all_scores.append(scores.cpu().numpy())
                all_labels.append(batch_dict["label"].cpu().numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        return scores, labels

    def compute_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            scores: Prediction scores.
            labels: True labels.
            threshold: Classification threshold.

        Returns:
            Dictionary of metrics.
        """
        # Binary predictions
        predictions = (scores >= threshold).astype(int)

        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )

        # Compute AUC if possible
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "threshold": float(threshold),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }

        return metrics

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.

        Args:
            dataloader: DataLoader for evaluation.
            return_predictions: Whether to return predictions.

        Returns:
            Dictionary containing evaluation results.
        """
        logger.info("Starting evaluation...")

        # Get predictions
        scores, labels = self.predict(dataloader)

        # Compute metrics for different thresholds
        results = {
            "scores": scores if return_predictions else None,
            "labels": labels if return_predictions else None,
            "metrics_by_threshold": {}
        }

        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in self.thresholds:
            metrics = self.compute_metrics(scores, labels, threshold)
            results["metrics_by_threshold"][threshold] = metrics

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold

        # Set best threshold metrics
        results["best_threshold"] = best_threshold
        results["best_metrics"] = results["metrics_by_threshold"][best_threshold]

        # Overall statistics
        results["num_samples"] = len(labels)
        results["num_positives"] = int(np.sum(labels == 1))
        results["num_negatives"] = int(np.sum(labels == 0))

        logger.info(f"Evaluation completed - Best F1: {best_f1:.4f} at threshold {best_threshold}")

        return results

    def evaluate_cross_dataset(
        self,
        dataloaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate on multiple datasets.

        Args:
            dataloaders: Dictionary of dataset name to DataLoader.

        Returns:
            Dictionary of dataset name to evaluation results.
        """
        results = {}

        for dataset_name, dataloader in dataloaders.items():
            logger.info(f"Evaluating on {dataset_name}...")
            results[dataset_name] = self.evaluate(dataloader)

        return results

    def compute_answerable_unanswerable_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute separate metrics for answerable and unanswerable questions.

        Args:
            scores: Prediction scores.
            labels: True labels.
            threshold: Classification threshold.

        Returns:
            Dictionary of metrics split by question type.
        """
        predictions = (scores >= threshold).astype(int)

        # Answerable (label = 1)
        answerable_mask = labels == 1
        if answerable_mask.sum() > 0:
            answerable_accuracy = accuracy_score(
                labels[answerable_mask],
                predictions[answerable_mask]
            )
        else:
            answerable_accuracy = 0.0

        # Unanswerable (label = 0)
        unanswerable_mask = labels == 0
        if unanswerable_mask.sum() > 0:
            unanswerable_accuracy = accuracy_score(
                labels[unanswerable_mask],
                predictions[unanswerable_mask]
            )
        else:
            unanswerable_accuracy = 0.0

        return {
            "answerable_accuracy": float(answerable_accuracy),
            "unanswerable_accuracy": float(unanswerable_accuracy),
            "num_answerable": int(answerable_mask.sum()),
            "num_unanswerable": int(unanswerable_mask.sum())
        }

    def save_predictions(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        output_path: str
    ) -> None:
        """Save predictions to file.

        Args:
            scores: Prediction scores.
            labels: True labels.
            output_path: Path to save predictions.
        """
        import json
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        predictions_data = {
            "scores": scores.tolist(),
            "labels": labels.tolist()
        }

        with open(output_file, "w") as f:
            json.dump(predictions_data, f, indent=2)

        logger.info(f"Saved predictions to {output_path}")
