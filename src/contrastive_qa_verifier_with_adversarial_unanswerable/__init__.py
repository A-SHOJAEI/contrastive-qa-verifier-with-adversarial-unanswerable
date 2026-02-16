"""
Contrastive QA Verifier with Adversarial Unanswerable Detection.

A dual-encoder system for verifying question-answer pair validity through
contrastive learning with adversarial training on unanswerable questions.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"
__license__ = "MIT"

from contrastive_qa_verifier_with_adversarial_unanswerable.models.model import (
    ContrastiveQAVerifier,
)
from contrastive_qa_verifier_with_adversarial_unanswerable.training.trainer import (
    QAVerifierTrainer,
)

__all__ = [
    "ContrastiveQAVerifier",
    "QAVerifierTrainer",
]
