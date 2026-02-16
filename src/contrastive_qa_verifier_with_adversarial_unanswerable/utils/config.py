"""Configuration utilities for the QA verifier."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the configuration.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {output_path}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration dictionary with override values.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested value from configuration using dot-separated key path.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to nested key (e.g., 'model.embedding_dim').
        default: Default value if key not found.

    Returns:
        Value at the specified path, or default if not found.
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_dir: Directory to store log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / "training.log"),
            logging.StreamHandler()
        ]
    )
