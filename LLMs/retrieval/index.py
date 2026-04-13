import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from config.settings import Config


def build_index(cfg: Config):
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")
    ds       = load_dataset(cfg.hf_dataset, split="train", token=cfg.hf_token)

    definitions = [r[cfg.definition_col] for r in ds]
    labels      = [r[cfg.label_col]      for r in ds]

    print("Embedding training definitions...")
    vectors = embedder.encode(definitions, normalize_embeddings=True, show_progress_bar=True)

    client     = chromadb.PersistentClient(path=cfg.index_path)
    collection = client.get_or_create_collection("definitions")

    # ChromaDB expects lists, and IDs must be strings
    collection.add(
        ids        = [str(i) for i in range(len(definitions))],
        embeddings = [v.tolist() for v in vectors],
        documents  = definitions,
        metadatas  = [{"label": label} for label in labels],
    )

    print(f"Index saved to {cfg.index_path} ({len(definitions)} entries)")
    