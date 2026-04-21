"""Configuration management for the polypwas package."""

import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "Rscript": "Rscript",
    "plink2": "plink2",
}

_config = None


def get_config_path() -> Path:
    """Return the user config path."""
    return Path.home() / ".polypwas" / "config.yaml"


def write_config(*, rscript: str, plink2: str | None = None) -> Path:
    """Write the user config file and refresh the in-process cache."""
    global _config
    config_path = get_config_path()
    config_path.parent.mkdir(exist_ok=True, parents=True)
    config = {**DEFAULT_CONFIG, "Rscript": rscript}
    if plink2 is not None:
        config["plink2"] = plink2
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    _config = config
    return config_path


def get_config(key=None):
    """Load and return the configuration.

    Reads from ~/.polypwas/config.yaml, falling back to defaults.

    Parameters
    ----------
    key : str, optional
        If provided, returns the value for this specific config key.

    Returns
    -------
    dict or any
        Configuration dictionary or specific config value if key is provided.
    """
    global _config
    if _config is None:
        config_path = get_config_path()
        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = yaml.safe_load(f) or {}
                _config = {**DEFAULT_CONFIG, **user_config}
            except Exception as e:
                logger.warning(f"Error loading config file: {e}. Using defaults.")
                _config = DEFAULT_CONFIG.copy()
        else:
            config_path.parent.mkdir(exist_ok=True, parents=True)
            with open(config_path, "w") as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            logger.info(f"Created default configuration file at {config_path}")
            _config = DEFAULT_CONFIG.copy()

    return _config[key] if key is not None else _config
