import mlx_lm
from .base import BaseModel


class QwenModel(BaseModel):
    """
    Qwen model loaded locally via MLX.
    """

    def __init__(self, model_path: str):
        print(f"Loading Qwen from {model_path} ...")
        self.model, self.tokenizer = mlx_lm.load(model_path)
        print("Qwen ready.")

    def query(self, definition: str, examples: str) -> str:
        prompt = self.build_prompt(definition, examples)
        
        return mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=False,
        )