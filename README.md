# Arabic Reverse Dictionary
 
> Given an Arabic definition or description, predict the word it describes.
 
Arabic was chosen intentionally — its rich morphology makes it one of the most challenging languages for NLP, and it remains comparatively underserved in the research space.
 
---
 
## Results (TL;DR)
 
| Approach | Best Top-1 (Overall) | Notes |
| :--- | :---: | :--- |
| **TF-IDF** | 18.18% | Strong keyword baseline; surprisingly competitive on short glosses |
| **FastText + FAISS** | 15.04% | Semantic search hurt by mean-pooling diluting rare words |
| **Transformers (Zero-Shot)** | 14.84% | CamelBERT best out-of-the-box |
| **Transformers (Fine-Tuned)** | 27.59% | Contrastive training with NT-Xent loss; CamelBERT leads |
| **Qwen3.5 Zero-Shot (LLM)** | 26.25%* | Morphological matching; 10.50% raw |
| **Qwen3.5 + RAG (LLM)** | **39.82%*** | Morphological matching; best result overall |
 
*\*Evaluated on a 1,000-sample subset due to hardware constraints.*
 
For full methodology, architecture details, loss functions, and per-model breakdowns → [read the blog post](https://amrbr.github.io/writing/reversedictionary/).
 
---
 
## Dataset
 
Data is aggregated from two sources into a final dataset of **97,822 entries**:
 
| Source | Split | Count |
| :--- | :--- | :--- |
| [KSAA-CAD](https://arai.ksaa.gov.sa/sharedTask/) | Train / Val / Test | 31,372 / 3,921 / 3,922 |
| [riotu-lab/arabic_reverse_dictionary](https://huggingface.co/datasets/riotu-lab/arabic_reverse_dictionary) | Train | 58,607 |
 
After preprocessing and merging, the final splits are:
 
| Split | Samples | Unique Words |
| :--- | :--- | :--- |
| **Train** | 76,265 | 35,310 |
| **Validation** | 9,533 | 7,201 |
| **Test** | 9,534 | 7,205 |
 
**Zero leakage**: 0.00% of test word-gloss pairs appear in training data.
 
---
 
## Project Structure
 
```
/config
    settings.py          # Environment variables and model hyperparameters
/data
    loader.py            # Dataset streaming and progress tracking (checkpointing)
/evaluation
    metrics.py           # Top-1, Top-5, and MRR implementations
    parser.py            # Regex-based output extraction for structured LLM results
/models
    base.py              # Abstract base class for model consistency
    gemma.py             # OpenAI-compatible API wrapper for Gemma
    qwen.py              # Native MLX implementation for Qwen
/retrieval
    index.py             # Vector database management (ChromaDB)
    retriever.py         # Candidate retrieval logic for RAG
Reverse_Dictionary.ipynb # The TF-IDF and Embeddings Experiments
 
main.py                  # Orchestration of the full LLM evaluation pipeline
```
 
---
 
## Models Evaluated
 
### Transformers
| Model | HuggingFace ID |
| :--- | :--- |
| Arabic-BERT | `asafaya/bert-base-arabic` |
| AraElectra | `aubmindlab/araelectra-base-discriminator` |
| AraBERT v2 | `aubmindlab/bert-base-arabertv2` |
| CamelBERT | `CAMeL-Lab/bert-base-arabic-camelbert-msa` |
| MARBERT | `UBC-NLP/MARBERT` |
| MARBERTv2 | `UBC-NLP/MARBERTv2` |
 
### LLMs (Local Inference)
| Model | Format | Runtime |
| :--- | :--- | :--- |
| Gemma-4-E4B | GGUF (Q4_K_M) | LM Studio API |
| Qwen3.5-4B | MLX (8-bit) | MLX-LM |
 
---
 
## Evaluation Metrics
 
- **Top-1 Accuracy** — Correct word is the model's first prediction
- **Top-5 Accuracy** — Correct word appears in the top 5 predictions
- **MRR (Mean Reciprocal Rank)** — Average reciprocal rank of the correct answer
For LLM evaluation, both **raw matching** and **morphological matching** are reported. Morphological matching uses CAMeL Tools for orthographic normalization, diacritic removal, article stripping, lemmatization, and root extraction — preventing penalization for morphologically valid answer variants.
 
---
 
## Key Takeaways
 
- TF-IDF outperformed static embeddings because **keyword importance matters more than broad semantics** for short dictionary glosses.
- Fine-tuning Transformers with contrastive learning **roughly doubled zero-shot performance** across all models.
- LLMs with RAG achieved the best results **without any fine-tuning**, and uniquely handle out-of-vocabulary (OOV) words that all retrieval-based methods fail on entirely.
- Arabic morphology makes exact-match evaluation misleading — morphological normalization revealed Qwen's true performance was **~2.5× higher** than raw matching suggested.