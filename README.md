# **Reverse-Dictionary-ML**

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

1.  **Normalization:** Removed metadata columns (id, pos, electra, bertseg, bertmsa) from the KSAA dataset to match the word/definition structure of the second source.
2.  **Text Cleaning:**
    * Handled null values by converting them to empty strings.
    * **De-diacritization:** Removed all diacritics/tashkeel.
    * **Tatweel Removal:** Removed the "ـ" character used for elongation.
    * **Orthographic Normalization:** Unified Alef (أ, إ, آ to ا), Alef Maksura (ى to ي), and Teh Marbuta (ة to ه).
3.  **Deduplication & Merging:** Removed duplicates within each set before merging them into a final dataset of 97,822 entries.
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
To maintain the integrity of the evaluation, we verified that no identical word-gloss pairs exist across different splits:
* **Word Overlap:** There are 4,060 words shared between Train and Validation, and 4,072 words shared between Train and Test. This is intentional to test if the model can identify the same word through different descriptive glosses.
* **Full Pair Overlap:** Both Train ∩ Val and Train ∩ Test result in **0**.
* **Critical Leakage:** 0.00% of Test pairs are present in the Training set.

### **Data Integrity Insights**
* **Multiple Glosses:** The dataset contains many instances where one word is mapped to multiple definitions, which helps the model learn diverse semantic contexts.
    * **Train:** 40,955 cases.
    * **Validation:** 2,332 cases.
    * **Test:** 2,329 cases.
* **Word Length Frequency (Training Set):** An analysis of the character length of target words and their frequency in the training data:
    `{1: 56142, 2: 15476, 3: 3638, 4: 807, 5: 153, 6: 36, 7: 8, 8: 3, 9: 1, 13: 1}`

---

## **Experimentation: TF-IDF**

The first phase of the project utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)**, a classical statistical method used to evaluate how important a word is to a document (in this case, a gloss) within the entire collection.

### **Methodology**
The system creates a vocabulary from the training data to build a sparse matrix.
* **Term Frequency (TF):** Calculates the frequency of a term in a specific gloss, often normalized by the gloss length.
    $$TF(t,d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$
* **Inverse Document Frequency (IDF):** Measures the importance of a term across all documents, assigning higher weights to rare words and lower weights to common terms.
    $$IDF(t,D) = \log\left(\frac{\text{Total number of documents } N}{\text{Number of documents with term } t \text{ in them}}\right)$$
* **Inference:** The test set is converted into a sparse matrix using the training vocabulary. Each row is then compared against the training matrix using **Cosine Similarity** to find the most relevant gloss, which is then mapped back to its target word.

### **Vectorizer Configuration**
The `scikit-learn` TfidfVectorizer was implemented with the following parameters to optimize results and manage memory:
* **analyzer="word"**: Operates on full words rather than character n-grams.
* **ngram_range=(1, 1)**: Considers only single words (unigrams).
* **min_df=2**: Words must appear at least twice to be considered, reducing noise from rare errors.
* **max_df=0.95**: Ignores terms appearing in more than 95% of documents to filter out overly common words.
* **max_features=30000**: Limits the vocabulary to the top 30,000 most frequent words to prevent excessive memory usage.
* **sublinear_tf=True**: Replaces raw TF with $1 + \log(TF)$ to scale down the impact of high-frequency words.
* **dtype=np.float32**: Uses 32-bit floats instead of 64-bit to save significant memory.

### **Evaluation Metrics**
1.  **Top-1:** The percentage of cases where the correct word is the model's first guess.
2.  **Top-5:** The percentage of cases where the correct word appears within the top 5 guesses.
3.  **MRR (Mean Reciprocal Rank):** The average of the reciprocal ranks for the correct words, giving more credit for answers appearing closer to the top.

### **Results (TF-IDF)**
With the configuration above, the model achieved the following baseline results:

| Metric | Result |
| :--- | :--- |
| **Top-1 Accuracy** | **0.1818** |
| **Top-5 Accuracy** | **0.2840** |
| **MRR (Mean Reciprocal Rank)** | **0.2205** |

---

## **Experiment 2: Static Embeddings (FastText + FAISS)**

In this phase, we move beyond keyword matching to **Semantic Search**. By using pre-trained vectors, we aim to find the "mathematical meaning" of a definition, allowing the system to recognize synonyms even if the exact words don't overlap between the query and the training set.

---

### **Methodology**
* **Model Selection:** We used the **FastText** Arabic model (`cc.ar.300.bin`), pre-trained on the Common Crawl dataset (2 million words/n-grams).
* **Subword Logic:** FastText was specifically chosen for its ability to handle Arabic’s rich morphology. By breaking words into **n-grams**, it can generate meaningful vectors for unseen words based on their roots and patterns.
* **Vectorization:** Each gloss (definition) was converted into a **300-dimensional vector**.
    * We applied **Mean Pooling** (averaging) to combine individual word vectors into a single "Sentence Vector" representing the entire definition.
* **Vector Database (FAISS):** To optimize memory and search speed, we implemented **FAISS** (Facebook AI Similarity Search).
    * We used an `IndexFlatIP` (Inner Product) index. 
    * By applying **L2 Normalization** to our vectors, the Inner Product calculation becomes mathematically equivalent to **Cosine Similarity**:
      $$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$



---

### **Results**
Despite the increased linguistic intelligence of the model, the initial scores were lower than the TF-IDF baseline:

| Metric | Result |
| :--- | :--- |
| **Top-1 Accuracy** | **0.1504** |
| **Top-5 Accuracy** | **0.2312** |
| **MRR (Mean Reciprocal Rank)** | **0.1817** |

---

### **Key Technical Insights & Observations**
* **The "Averaging" Weakness:** Unlike TF-IDF, which automatically rewards rare/important words via the IDF (Inverse Document Frequency) score, simple averaging in FastText treats all words equally. High-frequency words (e.g., "هو", "الذي", "في") pull the final vector toward a "generic" center, reducing the distinctiveness of the definition.

* This appears to be a common phenomenon in "Reverse Dictionaries" where short definitions are easily diluted by common functional words.

* The drop in performance compared to TF-IDF ($0.18$ vs $0.15$) suggests that for short, precise dictionary glosses, **keyword importance** is currently more valuable than **broad semantic meaning**. 


---

