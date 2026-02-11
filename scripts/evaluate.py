"""Evaluation script for Contrastive QA Verifier."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from contrastive_qa_verifier_with_adversarial_unanswerable.data.loader import (
    QADataLoader,
    load_squad_v2
)
from contrastive_qa_verifier_with_adversarial_unanswerable.data.preprocessing import (
    QAPreprocessor
)
from contrastive_qa_verifier_with_adversarial_unanswerable.models.model import (
    ContrastiveQAVerifier
)
from contrastive_qa_verifier_with_adversarial_unanswerable.evaluation.metrics import (
    QAVerifierEvaluator
)
from contrastive_qa_verifier_with_adversarial_unanswerable.utils.config import (
    load_config,
    setup_logging
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Contrastive QA Verifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def prepare_data(config, tokenizer, split="validation"):
    """Prepare evaluation data.

    Args:
        config: Configuration dictionary.
        tokenizer: Tokenizer instance.
        split: Dataset split to load.

    Returns:
        DataLoader for evaluation.
    """
    logger.info(f"Loading {split} dataset...")

    # Load SQuAD 2.0
    data_config = config.get("data", {})
    cache_dir = data_config.get("cache_dir", "./data/cache")

    dataset = load_squad_v2(split, max_samples=5000, cache_dir=cache_dir)

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = QAPreprocessor(config)
    questions, answers, contexts, labels = preprocessor.process_squad_v2(dataset)

    logger.info(f"Loaded {len(questions)} samples")

    # Create dataloader
    data_loader = QADataLoader(config, tokenizer)
    test_data = (questions, answers, contexts, labels)

    dataloaders = data_loader.prepare_dataloaders(
        test_data,  # Use as train (won't be used)
        test_data,  # Use as val
        test_data   # Use as test
    )

    return dataloaders["test"]


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config
    if args.device:
        config["system"]["device"] = args.device

    # Setup logging
    setup_logging("./logs")

    logger.info("Starting evaluation pipeline...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Device setup
    device_name = config.get("system", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    model_config = config.get("model", {})
    tokenizer_name = model_config.get("question_encoder", "sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Prepare data
    dataloader = prepare_data(config, tokenizer, args.split)

    # Initialize model
    logger.info("Initializing model...")
    model = ContrastiveQAVerifier(config)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    model_file = checkpoint_path / "model.pt" if checkpoint_path.is_dir() else checkpoint_path
    model.load_state_dict(torch.load(model_file, map_location=device))
    logger.info("Checkpoint loaded successfully")

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = QAVerifierEvaluator(model, config, device)

    # Evaluate
    logger.info("Running evaluation...")
    results = evaluator.evaluate(dataloader, return_predictions=args.save_predictions)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    best_metrics = results["best_metrics"]
    logger.info(f"Best Threshold: {results['best_threshold']:.2f}")
    logger.info(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {best_metrics['precision']:.4f}")
    logger.info(f"Recall:    {best_metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {best_metrics['f1']:.4f}")
    logger.info(f"AUC:       {best_metrics['auc']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TP: {best_metrics['true_positives']:5d}  FP: {best_metrics['false_positives']:5d}")
    logger.info(f"  FN: {best_metrics['false_negatives']:5d}  TN: {best_metrics['true_negatives']:5d}")
    logger.info("\nDataset Statistics:")
    logger.info(f"  Total samples:  {results['num_samples']}")
    logger.info(f"  Positives:      {results['num_positives']}")
    logger.info(f"  Negatives:      {results['num_negatives']}")
    logger.info("=" * 60 + "\n")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Remove numpy arrays for JSON serialization
        results_to_save = {
            k: v for k, v in results.items()
            if k not in ["scores", "labels"]
        }
        json.dump(results_to_save, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Save predictions if requested
    if args.save_predictions and results["scores"] is not None:
        predictions_file = output_dir / "predictions.json"
        evaluator.save_predictions(
            results["scores"],
            results["labels"],
            str(predictions_file)
        )
        logger.info(f"Predictions saved to: {predictions_file}")

    # Print metrics for all thresholds
    logger.info("\nMetrics by threshold:")
    logger.info("-" * 60)
    logger.info(f"{'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    logger.info("-" * 60)

    for threshold, metrics in sorted(results["metrics_by_threshold"].items()):
        logger.info(
            f"{threshold:<12.2f} {metrics['accuracy']:<10.4f} "
            f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
            f"{metrics['f1']:<10.4f}"
        )

    logger.info("-" * 60)
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
