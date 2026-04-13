from dataclasses import dataclass

@dataclass
class Config:
    hf_dataset: str = "Abreekaa/arabic-reverse-dictionary-merged"
    hf_split: str = "test"
    hf_token: str = None         
    definition_col: str = "gloss"
    label_col: str = "word"
    model_choice: str = "qwen"    
    results_file: str = "data/results.jsonl"
    mlx_model_path: str = "mlx-community/Qwen3.5-4B-MLX-8bit"
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "google/gemma-4-e4b"
    use_rag: bool = False
    index_path: str = "data/chroma_index"
    rag_top_k:  int = 3