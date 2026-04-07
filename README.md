# **Reverse-Dictionary**

The Reverse Dictionary project aims to build a tool that accepts Arabic descriptions or meanings as input and returns the most relevant matching words. This project serves as a comprehensive way to refresh Machine Learning and Natural Language Processing (NLP) knowledge, moving from classical statistical methods to modern architectures. The Arabic domain was specifically chosen due to its linguistic richness and the unique challenges presented by its underdevelopment in the NLP space.

## **Solution Overview**

The pipeline follows a structured approach: data collection from multiple sources, analysis, cleaning/normalization, and iterative experimentation with various algorithms including TF-IDF, embeddings, and eventually LLMs with RAG.

---

## **Data Collection**

Data is aggregated from two primary sources to create a robust dataset for training and evaluation.

### **1. KSAA-CAD: Contemporary Arabic Reverse Dictionary**
This dataset is derived from three major dictionaries: the "Contemporary Arabic Language Dictionary," "Mu'jam Arriyadh," and "Al Wassit LMF".
* **Format:** Data includes word lemmas, glosses (definitions), and Part of Speech (POS).
* **Linguistic Basis:** These dictionaries are based on lemmas rather than roots.
* **Contextual Embeddings:** The source also provides pre-computed embeddings from models like AraELECTRA, AraBERTv2, and camelBERT-MSA.
* **Example Entry:**
  ```json
  {
    "id": "ar.45",
    "word": "عين",
    "gloss": "عضو الإبصار في ...",
    "pos": "n",
    "electra":[0.4, 0.3, …],
    "bertseg":[0.7, 2.9, …],
    "bertmsa":[0.8, 1.4, …],
  }
* **Count:**
  - **Train:** 31372
  - **Validation:** 3921
  - **Test:** 3922
### **2. riotu-lab/arabic_reverse_dictionary**
A secondary dataset sourced from Hugging Face, provided in CSV format with two main columns: `word` and `definition`.
* **Count:**
  - **Train:** 58607

---

## **Data Preprocessing**

To ensure consistency across sources, the following pipeline was implemented using CamelTools:

1.  **Normalization:** Metadata columns (id, pos, electra, bertseg, bertmsa) were removed from the KSAA dataset to match the word/definition structure of the second source.
2.  **Text Cleaning:**
    * Null values were handled by conversion to empty strings.
    * **De-diacritization:** Removal of all diacritics/tashkeel.
    * **Tatweel Removal:** Removal of the "ـ" character used for elongation.
    * **Orthographic Normalization:** Unification of Alef (أ, إ, آ to ا), Alef Maksura (ى to ي), and Teh Marbuta (ة to ه).
3.  **Deduplication & Merging:** Duplicates within each set were removed before merging into a final dataset of 97,822 entries.
4.  **Splitting:** The final data was shuffled and split using an 80/10/10 ratio for Training, Validation, and Testing.

---

## **Data Integrity and Overlap Analysis**

Before proceeding with experimentation, a detailed analysis was conducted to ensure the validity of the data splits and to check for potential leakage between the sets.

| Metric | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| **Total Samples** | 76,265 | 9,533 | 9,534 |
| **Unique Words** | 35,310 | 7,201 | 7,205 |
| **Unique Pairs (W+G)** | 76,265 | 9,533 | 9,534 |

### **Overlap Analysis (Leakage)**
To maintain the integrity of the evaluation, verification confirmed that no identical word-gloss pairs exist across different splits:
* **Word Overlap:** 4,060 words are shared between Train and Validation, and 4,072 unique words are shared between Train and Test, with a total of 6,340 words shared. This is intentional to test model identification of the same word through different descriptive glosses.
* **Full Pair Overlap:** Both Train ∩ Val and Train ∩ Test result in **0**.
* **Critical Leakage:** 0.00% of Test pairs are present in the Training set.

### **Data Integrity Insights**
* **Multiple Glosses:** The dataset contains many instances where one word is mapped to multiple definitions, assisting the model in learning diverse semantic contexts.
    * **Train:** 40,955 cases.
    * **Validation:** 2,332 cases.
    * **Test:** 2,329 cases.
* **Word Length Frequency (Training Set):** Analysis of the word length of target words and their frequency:
    `{1: 56142, 2: 15476, 3: 3638, 4: 807, 5: 153, 6: 36, 7: 8, 8: 3, 9: 1, 13: 1}`

---

## **Experimentation: TF-IDF**

The first phase of the project utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)**, a classical statistical method used to evaluate the importance of a word to a document (gloss) within the entire collection.

### **Methodology**
The system creates a vocabulary from the training data to build a sparse matrix.
* **Term Frequency (TF):** Calculates the frequency of a term in a specific gloss, often normalized by the gloss length.
    $$TF(t,d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$
* **Inverse Document Frequency (IDF):** Measures the importance of a term across all documents, assigning higher weights to rare words and lower weights to common terms.
    $$IDF(t,D) = \log\left(\frac{\text{Total number of documents } N}{\text{Number of documents with term } t \text{ in them}}\right)$$
* **Inference:** The test set is converted into a sparse matrix using the training vocabulary. Each row is compared against the training matrix using **Cosine Similarity** to find the most relevant gloss, which is then mapped back to its target word.

### **Vectorizer Configuration**
The `scikit-learn` TfidfVectorizer was implemented with specific parameters to optimize results and manage memory:
* **analyzer="word"**: Operations performed on full words rather than character n-grams.
* **ngram_range=(1, 1)**: Consideration of single words (unigrams) only.
* **min_df=2**: Words must appear at least twice to be considered, reducing noise from rare errors.
* **max_df=0.95**: Ignore terms appearing in more than 95% of documents to filter out overly common words.
* **max_features=30000**: Vocabulary limited to the top 30,000 most frequent words to prevent excessive memory usage.
* **sublinear_tf=True**: Raw TF replaced with $1 + \log(TF)$ to scale down the impact of high-frequency words.
* **dtype=np.float32**: Use of 32-bit floats instead of 64-bit to save significant memory.

### **Evaluation Metrics**
1.  **Top-1:** The percentage of cases where the correct word is the model's first guess.
2.  **Top-5:** The percentage of cases where the correct word appears within the top 5 guesses.
3.  **MRR (Mean Reciprocal Rank):** Average of the reciprocal ranks for correct words, providing more credit for answers appearing closer to the top.

### **Results (TF-IDF)**
With the configuration above, the model achieved the following baseline results:

| Metric | Result |
| :--- | :--- |
| **Top-1 Accuracy** | **0.1818** |
| **Top-5 Accuracy** | **0.2840** |
| **MRR (Mean Reciprocal Rank)** | **0.2205** |

---

## **Experimentation: Static Embeddings (FastText + FAISS)**

In this phase, the approach moves beyond keyword matching to **Semantic Search**. By using pre-trained vectors, the goal is to find the mathematical meaning of a definition, allowing the system to recognize synonyms even without exact word overlap between the query and the training set.

### **Methodology**
* **Model Selection:** The **FastText** Arabic model (`cc.ar.300.bin`) was utilized, pre-trained on the Common Crawl dataset (2 million words/n-grams).
* **Subword Logic:** FastText was specifically chosen for its ability to handle Arabic’s rich morphology. By breaking words into **n-grams**, meaningful vectors can be generated for unseen words based on roots and patterns.
* **Vectorization:** Each gloss (definition) was converted into a **300-dimensional vector**.
    * **Mean Pooling** (averaging) was applied to combine individual word vectors into a single "Sentence Vector" representing the entire definition.
* **Vector Database (FAISS):** **FAISS** (Facebook AI Similarity Search) was implemented to optimize memory and search speed.
    * An `IndexFlatIP` (Inner Product) index was used. 
    * After applying **L2 Normalization** to the vectors, the Inner Product calculation becomes mathematically equivalent to **Cosine Similarity**:
      $$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

### **Results**
Despite the increased linguistic intelligence, initial scores were lower than the TF-IDF baseline:

| Metric | Result |
| :--- | :--- |
| **Top-1 Accuracy** | **0.1504** |
| **Top-5 Accuracy** | **0.2312** |
| **MRR (Mean Reciprocal Rank)** | **0.1817** |

### **Key Technical Insights & Observations**
* **The "Averaging" Weakness:** Unlike TF-IDF, which automatically rewards rare/important words via the IDF score, simple averaging in FastText treats all words equally. High-frequency words (e.g., "هو", "الذي", "في") pull the final vector toward a "generic" center, reducing the distinctiveness of the definition.
* This appears to be a common phenomenon in "Reverse Dictionaries" where short definitions are easily diluted by common functional words.
* The drop in performance compared to TF-IDF ($0.18$ vs $0.15$) suggests that for short, precise dictionary glosses, **keyword importance** is currently more valuable than **broad semantic meaning**. 

---

## **Experimentation: Dynamic Embeddings (Transformers)**

In this phase, the project moves from static word-lookup to **Contextual Attention Models**. Unlike previous experiments, Transformers analyze the entire sentence simultaneously, allowing the same word to have different mathematical representations based on its neighbors.

### **Execution: Zero-Shot vs. Fine-Tuning**

#### **Strategy 1: Zero-Shot (Baseline)**
The models are used "as-is" to encode the training and test datasets. **Cosine Similarity** is then used to find the closest match.
* **The In-Vocab Constraint:** Because the training set is a "Closed World" of 76k words, a specific metric is calculated for words existing in both sets. This identifies if failure stems from **Linguistic Logic** or if the **Word is New**.

#### **Strategy 2: Fine-Tuning (Alignment)**
Models will be re-trained using **Contrastive Learning**. This "forces" the model to move the vector of a definition closer to its specific target word while pushing it away from "distractor" words.

### **Model Selection**

Six Arabic models have been selected to compare "Out-of-the-box" understanding of dictionary glosses:

| Model | Source | HF |
| :--- | :--- | :--- |
| **Arabic-BERT** | Ali Safaya | asafaya/bert-base-arabic |
| **AraElectra** | AubMindLab | aubmindlab/araelectra-base-discriminator |
| **AraBERT v2** | AubMindLab | aubmindlab/bert-base-arabertv2 |
| **CamelBERT** | CAMeL-Lab | CAMeL-Lab/bert-base-arabic-camelbert-msa |
| **MARBERT** | UBC-NLP | UBC-NLP/MARBERT |
| **MARBERTv2** | UBC-NLP | UBC-NLP/MARBERTv2 |


### **Methodology (Zero-Shot Execution): The Transformer Pipeline**

The approach is divided into two strategies: **Zero-Shot Extraction** (using pre-trained knowledge) and **Supervised Fine-Tuning** (teaching the model the specific relationship between definitions and words).

#### **Stage A: Tokenization & Configuration**
The tokenizer breaks raw Arabic strings into "Sub-word" units. This is critical for Arabic, allowing the model to understand roots even with attached prefixes or suffixes.
* **Example:** The word "**وبالوالدين**" is tokenized into `['و', 'ب', 'ال', 'والدين']`.
* **Max Length (128):** A limit of 128 tokens was selected. 
    * *Too Small:* Truncates long dictionary definitions, losing unique word identifiers.
    * *Too Large:* Increases computational overhead and "dilutes" the vector with padding tokens.
* **Padding & Truncation:** Ensures all vectors in a batch have the same shape for GPU processing.

#### **Stage B: The Encoder Architecture**
Since the task is **Retrieval** (not generation), only the **Encoder** block is used. 
1.  **Self-Attention Head:** The "Core Engine" calculates a score for every token relative to every other token. This allows the model to realize that in the definition "بيت الشعر," the word "بيت" refers to a *verse*, not a *house*.
2.  **Feed-Forward Networks (MLP):** These layers process the attention-weighted information to extract higher-level semantic features.

#### **Stage C: Pooling Strategies**
After passing through 12+ layers, the model provides a vector for *every* token. To compare a query to the 76k definitions, these are "collapsed" into one **Sentence Vector**:
* **Mean Pooling:** Averaging all token vectors.
* **CLS Token:** Using the vector of the special `[CLS]` token designed to represent the whole sequence.
* **Max Pooling:** Taking the maximum value across all tokens for each dimension to highlight the most striking features.


### **Zero-Shot Results**

The following tables summarize the performance of various Transformer models using a **Zero-Shot** approach (Mean Pooling of raw embeddings) without any task-specific fine-tuning.

#### **1. Top-1 Accuracy**

| Model | In-Vocab Accuracy | Overall Accuracy |
| :--- | :---: | :---: |
| **CamelBERT** | **22.32%** | **14.84%** |
| **MARBERTv2** | 16.53% | 10.99% |
| **MARBERT** | 16.17% | 10.75% |
| **AraBERT** | 16.06% | 10.68% |
| **Arabic-BERT** | 16.03% | 10.66% |
| **AraElectra** | 10.09% | 6.71% |

---

#### **2. Top-5 Accuracy**

| Model | In-Vocab Accuracy | Overall Accuracy |
| :--- | :---: | :---: |
| **CamelBERT** | **37.56%** | **24.97%** |
| **MARBERTv2** | 27.52% | 18.30% |
| **MARBERT** | 26.18% | 17.41% | 
| **AraBERT** | 25.95% | 17.25% |
| **Arabic-BERT** | 25.93% | 17.24% |
| **AraElectra** | 15.95% | 10.60% |

---

#### **3. Mean Reciprocal Rank (MRR)**

| Model | In-Vocab MRR | Overall MRR |
| :--- | :---: | :---: |
| **CamelBERT** | **0.30** | **0.20** |
| **MARBERTv2** | 0.22 | 0.15 |
| **MARBERT** | 0.21 | 0.14 |
| **AraBERT** | 0.21 | 0.14 |
| **Arabic-BERT** | 0.21 | 0.14 |
| **AraElectra** | 0.13 | 0.09 |

---

### **Key Technical Insights & Observations**

* **The CamelBERT Advantage:** CamelBERT shows a nearly **6% lead** in Top-1 accuracy over its closest competitor (MARBERTv2). This suggests its internal "understanding" of formal Arabic vocabulary is superior for this specific dictionary task.
* **AraElectra Performance:** AraElectra performed significantly worse in Zero-Shot. This is expected, as Electra models are trained as "Discriminator" and often require fine-tuning to produce high-quality semantic embeddings for retrieval tasks.

---

### **Methodology (Fine Tuning): The Transformer Pipeline**
This section details the contrastive fine-tuning procedure applied to six state-of-the-art Arabic transformer models. The pipeline aligns gloss definitions with their target lexical items in a shared, semantically structured embedding space using a normalized temperature-scaled cross-entropy (NT-Xent) objective.

#### 1. Architectural Overview
Each fine-tuned model follows a two-stage architecture:
1. **Pre-trained Transformer Encoder**: Arabic-specific models (`Arabic-BERT`, `AraBERT`, `CamelBERT`, `MARBERT`, etc.) provide contextualized token representations.
2. **Task-Specific Projection Head**: A single linear layer maps the model's native hidden dimension to a compact `256`-dimensional space.

#### 2. Contrastive Learning Framework
Rather than treating word prediction as a classification problem, we frame it as a **dense retrieval alignment task**. For each training sample:
- A gloss `g` is pulled closer to its true target word embedding `w⁺`.
- The same gloss is pushed away from `N=5` randomly sampled distractor words `w₁⁻, ..., w₅⁻`.
- All embeddings are L2-normalized, meaning learning occurs on a **unit hypersphere** where cosine similarity equals dot product. This stabilizes gradients and improves metric compatibility.

#### 3. Loss Function: NT-Xent (InfoNCE)
The optimization objective is the **Normalized Temperature-scaled Cross-Entropy (NT-Xent)** loss, mathematically equivalent to the code's `F.cross_entropy(similarities / τ, targets=0)`:

For a batch of size `B` and `N` negatives per sample:

**1. Compute Similarities**:
```math
s_{\text{pos}} = \cos(z_g, z_{w^+}) \in \mathbb{R}^B
```

```math
s_{\text{neg}} = \left[ \cos(z_g, z_{w_1^-}), \dots, \cos(z_g, z_{w_N^-}) \right] \in \mathbb{R}^{B \times N}
```

**2. Scale by Temperature**:
```math
\text{logits} = \left[ \frac{s_{\text{pos}}}{\tau}, \frac{s_{\text{neg}}}{\tau} \right] \in \mathbb{R}^{B \times (1+N)}
```

**3. Cross-Entropy with Target 0**:
```math
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \left( \frac{\exp(s_{\text{pos}}^{(i)} / \tau)}{\exp(s_{\text{pos}}^{(i)} / \tau) + \sum_{j=1}^{N} \exp(s_{\text{neg}, j}^{(i)} / \tau)} \right)
```

**Why this works**: The loss rewards high positive similarity while penalizing high negative similarity. The denominator normalizes the score across all candidates, forcing the model to learn *relative* distances rather than absolute magnitudes.

#### 4. Negative Sampling Strategy
- **Count**: `negative_sample_size = 5` per gloss.
- **Sampling**: Negatives are drawn from the training vocabulary excluding the true target word.
- **Implementation**: Instead of encoding negatives sequentially, all `B × N` negatives are **flattened into a single batch**, tokenized, and passed through the transformer in one forward pass. The resulting embeddings are reshaped back to `(B, N, 256)` for loss computation.
- **Why 5?**: Empirically balances discriminative pressure and GPU memory. Fewer negatives yield weak gradients; more negatives increase compute without proportional gains in retrieval accuracy.

#### 5. Step-by-Step Epoch Workflow
Each training epoch processes the dataset in batches (`batch_size=128`) through the following pipeline:

| Step | Operation | Implementation Detail |
|------|-----------|------------------------|
| **1. Tokenization** | Convert glosses, true words, and negatives to `input_ids` + `attention_mask` | `max_length=128`, truncation + dynamic padding |
| **2. Forward Pass** | Encode all sequences through the transformer | Single pass for positives, batched pass for negatives |
| **3. Masked Mean Pooling** | Aggregate token states to sentence vectors | `∑(h_t · mask_t) / ∑(mask_t)`, ignoring padding tokens |
| **4. Projection & Norm** | Apply linear head + L2 normalization | `F.normalize(W·h + b, p=2, dim=1)` |
| **5. Similarity & Loss** | Compute NT-Xent loss | Temperature scaling → `cross_entropy` with target `0` |
| **6. Backward & Update** | Gradient computation & optimizer step | Mixed precision (FP16), gradient clipping (`max_norm=1.0`), AdamW |
| **7. LR Scheduling** | Update learning rate | Linear warmup (`500` steps) → linear decay to `0` |

#### 6. Hyperparameter Configuration & Rationale
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | `128` | Provides stable gradient estimates; maximizes GPU utilization for batched negative encoding |
| `learning_rate` | `2e-5` | Standard fine-tuning rate for BERT-family models; prevents catastrophic forgetting |
| `num_epochs` | `3` | Sufficient for convergence on gloss-word alignment without overfitting |
| `warmup_steps` | `500` | Stabilizes early updates when embedding similarities are noisy |
| `max_length` | `128` | Captures full Arabic glosses (~95th percentile length) without excessive padding |
| `temperature` | `0.07` | Sharpens softmax distribution; empirically optimal for sentence-level contrastive learning |
| `negative_sample_size` | `5` | Balances contrastive pressure, memory footprint, and training throughput |

#### 7. Optimization & Training Stability
To ensure robust convergence across diverse Arabic model architectures, the pipeline incorporates several industry-standard stabilization techniques:
- **AdamW Optimizer**: Decouples weight decay from gradient updates, applied only to the base transformer (projection head uses `weight_decay=0.0`).
- **Gradient Clipping**: `clip_grad_norm_(..., max_norm=1.0)` prevents exploding gradients.
- **Mixed Precision (AMP)**: `torch.autocast` + `GradScaler` accelerates training by ~2× and reduces VRAM usage without sacrificing numerical stability.
- **Projection Isolation**: The linear head is trained jointly but regularized separately, allowing the transformer to retain general Arabic semantics while adapting to gloss-word alignment.

#### 8. Evaluation Protocol (Post-Training)
After fine-tuning, models are evaluated via **gloss-to-word retrieval**:
1. All train/test glosses are embedded using the trained projection head.
2. Test gloss similarities are computed against all train word embeddings (L2-normalized cosine similarity).
3. Per-word scores are aggregated via **max-pooling** across multiple glosses per word.
4. Ranking metrics are computed: `Top-1`, `Top-5`, and `Mean Reciprocal Rank (MRR)`, reported separately for In-Vocabulary (IV) and Out-Of-Vocabulary (OOV) test samples.


### **Fine Tuning Results**
The following shows the fine-tuning loss per epoch for each model:
| Model | Epoch 1 | Epoch 2 | Epoch 3 | Average Epoch Duration(mins) | 
| :--- | :---: | :---: | :---: | :---: |
| **CamelBERT** | 0.3533 | 0.1509 | 0.1039 | 20:31,  1.03s/it |
| **MARBERTv2** | 1.5419 | 0.1987 | 0.1335 | 19:11  1.03it/s |
| **MARBERT** | 0.5900 | 0.1966 | 0.1146 | 19:11  1.03it/s |
| **AraBERT** | 0.7003 | 0.3246 | 0.2607 | 21:36,  2.18s/it |
| **Arabic-BERT** | 0.4569 | 0.1965 | 0.1187 | 20:59,  1.89it/s |
| **AraElectra** | 0.7004 | 0.2425 | 0.1658 | 21:17, 1.87it/s |

The following tables summarize the performance of the models after fine-tuning for the task.

#### **1. Top-1 Accuracy**

| Model | In-Vocab Accuracy | Overall Accuracy |
| :--- | :---: | :---: |
| **CamelBERT** | 41.48% | 27.59% |
| **MARBERTv2** | 40.60% | 27.00% |
| **MARBERT** | 38.85% | 25.83% |
| **AraBERT** | 38.20% | 25.40% |
| **Arabic-BERT** | 39.54% | 26.30% |
| **AraElectra** | 29.83%  | 19.83% |

---

#### **2. Top-5 Accuracy**

| Model | In-Vocab Accuracy | Overall Accuracy |
| :--- | :---: | :---: |
| **CamelBERT** | 59.78% | 39.75% |
| **MARBERTv2** | 58.41% | 38.84% |
| **MARBERT** | 56.34% | 37.47% |
| **AraBERT** | 53.85% | 35.81% |
| **Arabic-BERT** | 55.66% | 37.01% |
| **AraElectra** | 46.23% | 30.74% |

---

#### **3. Mean Reciprocal Rank (MRR)**

| Model | In-Vocab Accuracy | Overall Accuracy |
| :--- | :---: | :---: |
| **CamelBERT** | 0.50 | 0.33 |
| **MARBERTv2** | 0.49 | 0.32 |
| **MARBERT** | 0.47 | 0.31 |
| **AraBERT** | 0.46 | 0.30 |
| **Arabic-BERT** | 0.47 | 0.31 |
| **AraElectra** | 0.38 | 0.25 |

---

### **Key Technical Insights & Observations**

* **CamelBERT Performance:** It maintained its lead from the baseline, showing that its diverse pre-training on dialects and MSA provided the most stable foundation for this task.

* **The Equalization Effect:** Fine-tuning significantly closed the gap between models, showing that contrastive alignment can overcome initial architectural weaknesses.

* **AraElectra’s Recovery:** Despite poor zero-shot performance, its accuracy tripled after fine-tuning, confirming that discriminator models require task-specific training to produce useful embeddings.

* **Semantic Neighborhoods:** The high Top-5 accuracy suggests the models are effectively clustering synonyms and related concepts, even when they miss the exact target word.

