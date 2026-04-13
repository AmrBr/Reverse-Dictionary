from openai import OpenAI
from .base import BaseModel


class GemmaModel(BaseModel):
    """
    Gemma model served via LM Studio (OpenAI-compatible endpoint).
    Requires LM Studio running locally with a loaded model.
    """

    def __init__(self, base_url: str, model_name: str):
        print(f"Connecting to LM Studio at {base_url} (model: {model_name}) ...")
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model_name = model_name
        print("Gemma ready.")

    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        return response.choices[0].message.content