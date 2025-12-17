from pathlib import Path
import yaml


def load_config(config_path:str, **overrides) -> dict:
    """Load configuration from YAML file with optional CLI overrides
    
    Args:
        config_path (str): Path to the YAML configuration file (required)
        **overrides: CLI argument overrides in dot notation format
    
    Returns:
        dict: Loaded configuration with overrides applied
    
    Raises:
        ValueError: If config_path is None or empty
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None or config_path == "":
        raise ValueError("config_path is required and cannot be None or empty")
    
    if config_path.endswith(".yml") is False and config_path.endswith(".yaml") is False:
        raise ValueError("config_path must point to a YAML file with .yml or .yaml extension")
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Apply CLI overrides to nested config
    for key, value in overrides.items():
        if value is not None:  # Only override if value was provided
            keys = key.split('.')
            current = cfg
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    return cfg