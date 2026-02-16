"""Training script for Contrastive QA Verifier."""

import argparse
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
    QAPreprocessor,
    AdversarialGenerator
)
from contrastive_qa_verifier_with_adversarial_unanswerable.models.model import (
    ContrastiveQAVerifier
)
from contrastive_qa_verifier_with_adversarial_unanswerable.training.trainer import (
    QAVerifierTrainer
)
from contrastive_qa_verifier_with_adversarial_unanswerable.utils.config import (
    load_config,
    save_config,
    set_seed,
    setup_logging
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Contrastive QA Verifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def prepare_data(config, tokenizer):
    """Prepare training data.

    Args:
        config: Configuration dictionary.
        tokenizer: Tokenizer instance.

    Returns:
        Dictionary with train, val, and test dataloaders.
    """
    logger.info("Loading datasets...")

    # Load SQuAD 2.0
    data_config = config.get("data", {})
    datasets_config = config.get("datasets", {})
    cache_dir = data_config.get("cache_dir", "./data/cache")

    squad_config = datasets_config.get("squad_v2", {})
    max_samples = squad_config.get("max_samples", 50000)

    train_dataset = load_squad_v2("train", max_samples=max_samples, cache_dir=cache_dir)
    val_dataset = load_squad_v2("validation", max_samples=5000, cache_dir=cache_dir)

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = QAPreprocessor(config)

    train_questions, train_answers, train_contexts, train_labels = preprocessor.process_squad_v2(train_dataset)
    val_questions, val_answers, val_contexts, val_labels = preprocessor.process_squad_v2(val_dataset)

    # Generate adversarial examples
    logger.info("Generating adversarial examples...")
    adversarial_gen = AdversarialGenerator(config)

    train_questions, train_answers, train_contexts, train_labels = adversarial_gen.generate_adversarial_examples(
        train_questions, train_answers, train_contexts, train_labels
    )

    train_questions, train_answers, train_contexts, train_labels = adversarial_gen.swap_answers(
        train_questions, train_answers, train_contexts, train_labels
    )

    logger.info(f"Training samples: {len(train_questions)}")
    logger.info(f"Validation samples: {len(val_questions)}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    data_loader = QADataLoader(config, tokenizer)

    train_data = (train_questions, train_answers, train_contexts, train_labels)
    val_data = (val_questions, val_answers, val_contexts, val_labels)

    dataloaders = data_loader.prepare_dataloaders(train_data, val_data)

    return dataloaders


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.device:
        config["system"]["device"] = args.device

    # Setup logging
    log_dir = config.get("paths", {}).get("log_dir", "./logs")
    setup_logging(log_dir)

    logger.info("Starting training pipeline...")
    logger.info(f"Configuration: {args.config}")

    # Set random seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    # Save configuration
    output_dir = Path(config.get("paths", {}).get("output_dir", "./models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "config.yaml"))

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
    dataloaders = prepare_data(config, tokenizer)

    # Initialize model
    logger.info("Initializing model...")
    model = ContrastiveQAVerifier(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = QAVerifierTrainer(
        model=model,
        config=config,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        device=device
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed successfully!")
    logger.info(f"Best model saved to: {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()
