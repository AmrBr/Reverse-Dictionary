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

### **Methodology: The Transformer Pipeline**

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

### **Execution: Zero-Shot vs. Fine-Tuning**

#### **Strategy 1: Zero-Shot (Baseline)**
The models are used "as-is" to encode the training and test datasets. **Cosine Similarity** is then used to find the closest match.
* **The In-Vocab Constraint:** Because the training set is a "Closed World" of 76k words, a specific metric is calculated for words existing in both sets. This identifies if failure stems from **Linguistic Logic** or if the **Word is New**.

#### **Strategy 2: Fine-Tuning (Alignment)**
Models will be re-trained using **Contrastive Learning**. This "forces" the model to move the vector of a definition closer to its specific target word while pushing it away from "distractor" words.

### **Model Selection**

Five Arabic models have been selected to compare "Out-of-the-box" understanding of dictionary glosses:

| Model | Source |
| :--- | :--- |
| **Arabic-BERT** | Ali Safaya |
| **AraElectra** | AubMindLab |
| **AraBERT v2** | AubMindLab |
| **CamelBERT** | NYU Abu Dhabi |
| **MARBERT** | UBC-NLP |

### **Results**

### **Key Technical Insights & Observations**

