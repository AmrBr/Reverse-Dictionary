from config.settings import Config
from .base import BaseModel


def load_model(cfg: Config) -> BaseModel:
    """
    Factory: returns the correct model backend based on cfg.model_choice.
    Add new backends here — main.py never needs to change.
    """
    if cfg.model_choice == "qwen":
        from .qwen import QwenModel
        return QwenModel(model_path=cfg.mlx_model_path)

    if cfg.model_choice == "gemma":
        from .gemma import GemmaModel
        return GemmaModel(base_url=cfg.lm_studio_url, model_name=cfg.lm_studio_model)

    raise ValueError(
        f"Unknown model_choice '{cfg.model_choice}'. "
        "Expected 'qwen' or 'gemma'."
    )


__all__ = ["load_model", "BaseModel"]