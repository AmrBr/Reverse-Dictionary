import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from config.settings import Config


def build_index(cfg: Config):
    client     = chromadb.PersistentClient(path=cfg.index_path)
    collection = client.get_or_create_collection("definitions")
    ds       = load_dataset(cfg.hf_dataset, split="train", token=cfg.hf_token)
    
    expected = len(ds)
    
    if collection.count() == expected:
        print(f"Index already exists ({collection.count()} entries). Skipping build.")
        return

    # if partial, wipe and rebuild cleanly
    if collection.count() > 0:
        print(f"Partial index detected ({collection.count()}/{expected}). Rebuilding...")
        client.delete_collection("definitions")
        collection = client.get_or_create_collection("definitions")

    embedder = SentenceTransformer("intfloat/multilingual-e5-base")

    definitions = [r[cfg.definition_col] for r in ds]
    labels      = [r[cfg.label_col]      for r in ds]

    print("Embedding training definitions...")
    vectors = embedder.encode(
        definitions, normalize_embeddings=True, show_progress_bar=True
    )

    print("Adding to ChromaDB in batches...")
    for start in range(0, len(definitions), cfg.CHROMA_BATCH_SIZE):
        end = min(start + cfg.CHROMA_BATCH_SIZE, len(definitions))
        collection.add(
            ids        = [str(i) for i in range(start, end)],
            embeddings = [v.tolist() for v in vectors[start:end]],
            documents  = definitions[start:end],
            metadatas  = [{"label": label} for label in labels[start:end]],
        )
        print(f"  Added {end}/{len(definitions)}")

    print(f"Index built: {len(definitions)} entries saved to {cfg.index_path}")