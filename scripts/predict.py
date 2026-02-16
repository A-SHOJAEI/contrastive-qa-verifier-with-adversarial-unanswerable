"""Inference script for Contrastive QA Verifier."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from contrastive_qa_verifier_with_adversarial_unanswerable.models.model import (
    ContrastiveQAVerifier
)
from contrastive_qa_verifier_with_adversarial_unanswerable.utils.config import (
    load_config,
    setup_logging
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Contrastive QA Verifier")
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
        "--question",
        type=str,
        help="Question text"
    )
    parser.add_argument(
        "--answer",
        type=str,
        help="Answer text"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="JSON file with questions and answers"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for positive prediction"
    )
    return parser.parse_args()


def predict_single(model, tokenizer, question, answer, device, threshold=0.5):
    """Run prediction on a single Q-A pair.

    Args:
        model: The QA verifier model.
        tokenizer: Tokenizer instance.
        question: Question text.
        answer: Answer text.
        device: Device to run on.
        threshold: Similarity threshold.

    Returns:
        Dictionary with predictions.
    """
    # Tokenize
    q_enc = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    a_enc = tokenizer(
        answer,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Move to device
    q_enc = {k: v.to(device) for k, v in q_enc.items()}
    a_enc = {k: v.to(device) for k, v in a_enc.items()}

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(
            question_input_ids=q_enc["input_ids"],
            question_attention_mask=q_enc["attention_mask"],
            answer_input_ids=a_enc["input_ids"],
            answer_attention_mask=a_enc["attention_mask"],
        )

        # Get similarity score
        similarity = outputs["similarity"][0, 0].item()

        # Get answerability prediction
        answerable_logits = outputs["answerable_logits"][0]
        answerable_probs = torch.softmax(answerable_logits, dim=0)
        answerable_prob = answerable_probs[1].item()

        # Make prediction
        is_correct = similarity >= threshold
        is_answerable = answerable_prob >= 0.5

    return {
        "question": question,
        "answer": answer,
        "similarity_score": round(similarity, 4),
        "answerable_probability": round(answerable_prob, 4),
        "is_correct": bool(is_correct),
        "is_answerable": bool(is_answerable),
        "prediction": "correct" if is_correct and is_answerable else "incorrect" if is_answerable else "unanswerable"
    }


def predict_batch(model, tokenizer, qa_pairs, device, threshold=0.5):
    """Run prediction on a batch of Q-A pairs.

    Args:
        model: The QA verifier model.
        tokenizer: Tokenizer instance.
        qa_pairs: List of (question, answer) tuples.
        device: Device to run on.
        threshold: Similarity threshold.

    Returns:
        List of prediction dictionaries.
    """
    results = []
    for question, answer in qa_pairs:
        result = predict_single(model, tokenizer, question, answer, device, threshold)
        results.append(result)
    return results


def main():
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config
    if args.device:
        config["system"]["device"] = args.device

    # Setup logging
    setup_logging("./logs")

    logger.info("Starting prediction pipeline...")
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

    # Initialize model
    logger.info("Initializing model...")
    model = ContrastiveQAVerifier(config)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    model_file = checkpoint_path / "model.pt" if checkpoint_path.is_dir() else checkpoint_path
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    logger.info("Checkpoint loaded successfully")

    # Run prediction
    if args.input_file:
        # Batch prediction from file
        logger.info(f"Loading questions from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            data = json.load(f)

        qa_pairs = [(item["question"], item["answer"]) for item in data]
        logger.info(f"Running prediction on {len(qa_pairs)} Q-A pairs...")

        results = predict_batch(model, tokenizer, qa_pairs, device, args.threshold)

        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(json.dumps(results, indent=2))

    elif args.question and args.answer:
        # Single prediction
        logger.info("Running prediction on single Q-A pair...")
        result = predict_single(
            model, tokenizer, args.question, args.answer, device, args.threshold
        )

        # Print result
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION RESULT")
        logger.info("=" * 60)
        logger.info(f"Question: {result['question']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Similarity Score: {result['similarity_score']:.4f}")
        logger.info(f"Answerable Probability: {result['answerable_probability']:.4f}")
        logger.info(f"Is Correct: {result['is_correct']}")
        logger.info(f"Is Answerable: {result['is_answerable']}")
        logger.info(f"Prediction: {result['prediction']}")
        logger.info("=" * 60 + "\n")

        # Save if output file specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to {args.output_file}")
    else:
        logger.error("Either --question and --answer or --input_file must be provided")
        sys.exit(1)

    logger.info("Prediction completed successfully!")


if __name__ == "__main__":
    main()
