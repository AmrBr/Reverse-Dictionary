# Arabic Reverse Dictionary Project - Week-by-Week Execution Plan

This plan breaks the project into **manageable stages**, progressively covering old ML, modern NLP, embeddings, retrieval, RAG, and deployment. It is end-to-end and resume/interview-ready.

---

## Week 1-2: Data Collection & Cleaning

**Goals:**

* Gather high-quality Arabic dictionary data.
* Establish train/validation/test splits.
* Preprocess text for consistency.

**Tasks:**

* Collect data from Arabic dictionaries, WordNet, Wiktionary.
* Ensure each entry has: `word`, `definition(s)`, optional `examples`.
* Split dataset: ~70% train, 15% validation, 15% test.
* Cleaning steps:

  * Normalize Alef forms (ا, أ, إ)
  * Normalize Yeh/Teh (ي, ى)
  * Remove tashkeel/diacritics
  * Normalize punctuation and whitespace
* Optional: generate paraphrases or synonyms for queries (for evaluation and later robustness testing)

**Deliverables:**

* Cleaned dataset in JSON or CSV
* Initial train/val/test splits
* Preprocessing pipeline as reusable Python functions

---

## Week 3: Baseline Experiments (Classical NLP)

**Goals:**

* Establish a baseline for retrieval performance.

**Tasks:**

* Implement TF-IDF vectorizer on definitions.
* Compute cosine similarity for retrieval.
* Evaluate on test set: top-1, top-5 accuracy, MRR.
* Optional: classical ML approach (TF-IDF + Logistic Regression) to frame as classification problem.

**Deliverables:**

* Baseline retrieval performance table
* Analysis of failure cases and limitations

---

## Week 4-5: Pretrained Transformer Experiments

**Goals:**

* Evaluate existing models (MarBERT, AraBERT, or other Arabic embeddings) without fine-tuning.
* Compare embeddings retrieval vs cross-encoder scoring.

**Tasks:**

* Use pretrained models to encode definitions and queries.
* Compute similarity (cosine) for retrieval.
* Evaluate metrics: top-1/top-5 accuracy, MRR.
* Compare retrieval speed and memory usage.
* Document observations: strengths, weaknesses, semantic gaps.

**Deliverables:**

* Pretrained model performance metrics
* Comparison with TF-IDF baseline
* Notebook documenting experiments

---

## Week 6-7: Fine-Tuning Embeddings (Bi-Encoder)

**Goals:**

* Improve retrieval by fine-tuning embeddings on your dataset.

**Tasks:**

* Prepare positive pairs: `(query, correct word/definition)`
* Sample negative examples (random or hard negatives)
* Train bi-encoder with contrastive loss
* Evaluate on validation set, check top-k retrieval
* Compare with pretrained-only embeddings

**Deliverables:**

* Fine-tuned bi-encoder model
* Evaluation metrics and comparison charts
* Notebook documenting fine-tuning process

---

## Week 8: Vector Database Integration

**Goals:**

* Implement scalable retrieval using vector DB.
* Precompute embeddings and store in FAISS index.

**Tasks:**

* Embed all dictionary definitions offline
* Build FAISS index with vectors and metadata
* Implement query-time retrieval pipeline (load index, embed query, find top-k)
* Evaluate retrieval speed, latency, and accuracy
* Optional: rerank top-k with cross-encoder for accuracy improvement

**Deliverables:**

* FAISS index and retrieval code
* Performance benchmarks (latency vs accuracy)
* Updated notebook / scripts

**Note:** VectorDB alone is **retrieval**, not RAG. RAG requires an LLM in the pipeline.

---

## Week 9: Optional LLM Integration (RAG)

**Goals:**

* Use retrieval results to generate explanations or enhance user-facing output.

**Tasks:**

* Select a small-scale LLM (or API like OpenAI GPT) for text generation
* Feed top-k retrieved definitions to LLM
* Generate human-readable explanations or paraphrases
* Evaluate qualitative improvements

**Deliverables:**

* Example outputs for sample queries
* Qualitative evaluation report
* Documentation of integration

---

## Week 10-11: Evaluation & Comparative Analysis

**Goals:**

* Systematically compare all approaches.
* Analyze quantitative and qualitative performance.

**Tasks:**

* Compile metrics from TF-IDF, pretrained embeddings, fine-tuned embeddings, vectorDB retrieval, optional LLM
* Make comparative tables: accuracy, MRR, latency, memory
* Perform qualitative analysis: semantic errors, polysemy, rare words
* Document key insights: which methods improve what, trade-offs

**Deliverables:**

* Comprehensive comparison report
* Plots and tables for top-1/top-5 retrieval, latency, memory usage
* Failure analysis write-up

---

## Week 12: Deployment Pipeline

**Goals:**

* Build a fully functional, containerized service
* Separate offline and online pipelines

**Tasks:**

* Offline pipeline:

  * Data cleaning
  * Model training / fine-tuning
  * Embedding computation
  * FAISS index building
* Online pipeline:

  * Load fine-tuned model
  * Load FAISS index
  * Accept user queries via API (FastAPI)
  * Return top-k words (optional: LLM explanations)
* Dockerize the backend (model + API + FAISS)
* Optional: simple frontend (Streamlit / React) for demo
* Test latency and reliability

**Deliverables:**

* Dockerized deployment of reverse dictionary
* API documentation and usage examples
* Optional demo frontend

---

## Week 13: Code Organization & Cleanup

**Goals:**

* Make the repo professional and modular

**Tasks:**

* Convert notebooks to Python scripts / modules:

  * `data/` → cleaning, preprocessing
  * `training/` → baselines, fine-tuning, evaluation
  * `indexing/` → FAISS build & query
  * `serving/` → FastAPI app, model loader
  * `artifacts/` → model checkpoints, index files
  * `experiments/` → results, logs
* Add README, instructions, hyperparameter configs, and example usage
* Ensure reproducibility

**Deliverables:**

* Clean, modular repo ready for portfolio / GitHub
* Clear instructions for reproducing results

---

## Optional Advanced / Stretch Goals

* Adversarial / robustness testing (paraphrases, noisy queries)
* Multiple embedding models comparison (e.g., AraBERT vs MarBERT vs multilingual SBERT)
* Fine-tuning cross-encoder reranking
* Automatic metric logging and visualization

---

