import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import Config


class Retriever:
    def __init__(self, cfg: Config):
        self.embedder  = SentenceTransformer("intfloat/multilingual-e5-base")
        client         = chromadb.PersistentClient(path=cfg.index_path)
        self.collection = client.get_collection("definitions")
        self.top_k     = cfg.rag_top_k

    def augment(self, definition: str) -> str:
        """Embed the definition, retrieve similar examples, return augmented prompt."""
        vec = self.embedder.encode(definition, normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=self.top_k,
        )

        # Build few-shot prefix
        shots = "بعض الامثلة المشابهة:\n\n"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            shots += f"التعريف: {doc}\nالإجابة: {meta['label']}\n\n"

        return shots + f"التعريف: {definition}"