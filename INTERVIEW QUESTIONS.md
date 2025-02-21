
You have 10 documents. Each topic has been tagged. When a new document arrives, how to tag it?
## **Document Classification: How to Tag a New Document Based on Existing Topics?**  

When you have **10 documents**, each tagged with a topic, and a **new document arrives**, you need to classify it into one of the existing topics. This is known as **Text Classification** or **Document Categorization**. 

We can use multiple approaches, including **Machine Learning (ML), Deep Learning (DL), and Rule-Based Methods**. Let's break this down step by step.  

---

## **Step 1: Understanding the Problem**
- We have **10 labeled documents** (e.g., "Politics," "Technology," "Sports," etc.).
- A new document arrives.
- We need to assign a topic **automatically** based on similarity with existing documents.  

### **Example Dataset**
| Document | Content Snippet | Label (Topic) |
|----------|----------------|--------------|
| Doc 1 | "Government passes new law on taxation" | Politics |
| Doc 2 | "Apple releases new iPhone with AI features" | Technology |
| Doc 3 | "Real Madrid wins Champions League" | Sports |
| Doc 4 | "Python is a popular programming language" | Technology |
| Doc 5 | "NASA successfully lands rover on Mars" | Science |
| ... | ... | ... |
| New Document | "Tesla introduces autopilot update with improved AI" | ??? |

We need to classify the **new document** into an existing topic.

---

## **Step 2: Preprocessing the Documents**
Before we train a model, we need to convert raw text into a machine-readable format.

1. **Text Cleaning**
   - Convert to lowercase  
   - Remove stop words (e.g., "the", "is", "a")  
   - Remove punctuation and special characters  
   - Tokenization (split text into words)  
   - Lemmatization/Stemming (reduce words to their root form)

   **Example:**  
   `"Tesla introduces autopilot update with improved AI"`
   → **Preprocessed Text:** `["tesla", "introduce", "autopilot", "update", "improve", "ai"]`

2. **Feature Extraction (Vectorization)**
   - Convert text into numerical representation.
   - Common methods:
     - **Bag of Words (BoW)** (word frequency)
     - **TF-IDF (Term Frequency - Inverse Document Frequency)**
     - **Word Embeddings (Word2Vec, GloVe, BERT)**
     - **Sentence Embeddings (SBERT, Universal Sentence Encoder)**  

   **Example: TF-IDF Representation**  
   ```
   Document 1: [0.1, 0.3, 0.0, 0.5, ...]  
   Document 2: [0.2, 0.1, 0.6, 0.0, ...]  
   New Doc   : [0.0, 0.1, 0.4, 0.2, ...]  
   ```

---

## **Step 3: Choosing a Classification Approach**
There are **three** main approaches:

### **1. Similarity-Based Approach (KNN, Cosine Similarity)**
📌 **Best for small datasets (like our 10 documents).**  
- Calculate similarity between the new document and existing ones.  
- Use **Cosine Similarity** or **Euclidean Distance** to find the most similar document.  
- Assign the label of the most similar document.

📌 **Example: Cosine Similarity Calculation**  
\[
\text{Cosine Similarity} = \frac{A \cdot B}{||A|| \times ||B||}
\]
Where **A** and **B** are TF-IDF vectors of the documents.

📌 **Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "Government passes new law on taxation",
    "Apple releases new iPhone with AI features",
    "Real Madrid wins Champions League",
    "Python is a popular programming language",
    "NASA successfully lands rover on Mars"
]

labels = ["Politics", "Technology", "Sports", "Technology", "Science"]

# New document to classify
new_doc = ["Tesla introduces autopilot update with improved AI"]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents + new_doc)

# Compute similarity
similarities = cosine_similarity(X[-1], X[:-1])  # Compare new doc with existing ones
best_match_idx = similarities.argmax()
predicted_label = labels[best_match_idx]

print(f"Predicted Topic: {predicted_label}")  # Output: Technology
```

---

### **2. Machine Learning-Based Approach (Naïve Bayes, SVM, Random Forest)**
- Train an ML classifier on the existing labeled documents.
- Use feature vectors (TF-IDF, BoW) as input.
- Train on historical data and predict on new documents.

📌 **Steps:**  
1. **Train the model using labeled documents**  
2. **Convert new document to vector representation**  
3. **Predict label using the trained model**  

📌 **Implementation using Naïve Bayes:**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Convert documents to vectors
X = vectorizer.fit_transform(documents)
y = labels  # Labels

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Predict label for new document
new_X = vectorizer.transform(new_doc)
predicted_label = model.predict(new_X)[0]

print(f"Predicted Topic: {predicted_label}")  # Output: Technology
```

🔹 **Advantages:**  
✅ Works well for text classification  
✅ Fast and scalable  
✅ Handles noisy data well  

🔹 **Disadvantages:**  
❌ Requires enough training data  
❌ May not work well for unseen words  

---

### **3. Deep Learning Approach (LSTMs, Transformers like BERT)**
- Convert text into **word embeddings** (Word2Vec, GloVe, BERT).  
- Train a deep learning model (LSTM, CNN, Transformer).  
- Predict topic based on learned representations.  

📌 **Example using BERT for classification:**
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

new_text = "Tesla introduces autopilot update with improved AI"
labels = ["Politics", "Technology", "Sports", "Science"]

result = classifier(new_text, candidate_labels=labels)
print(result['labels'][0])  # Output: Technology
```
🔹 **Advantages:**  
✅ **Highly accurate**  
✅ Can understand **context & semantics**  
✅ Works well on large datasets  

🔹 **Disadvantages:**  
❌ Requires **large dataset & high computational power**  
❌ Training can be slow  

---

## **Step 4: Evaluating Model Performance**
- Use **Accuracy, Precision, Recall, and F1-Score** to measure performance.  
- Use a **confusion matrix** to analyze misclassifications.  

📌 **Evaluation Example:**
```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## **Step 5: Deploying the Model**
- Convert the trained model into an API (Flask/FastAPI).  
- Integrate it into a **real-world application** (e.g., search engine, chatbot).  
- Deploy on **cloud platforms** (AWS, GCP, Azure).  

---

## **Final Summary: Best Approach Based on Your Use Case**
| Approach | Best For | Pros | Cons |
|----------|---------|------|------|
| Cosine Similarity (KNN) | Small datasets | Simple, No training needed | Slow for large data |
| Naïve Bayes / SVM | Medium-sized datasets | Fast, Good accuracy | Needs labeled data |
| Deep Learning (BERT, LSTM) | Large datasets | Best accuracy, Understands semantics | Requires high compute |

---

## **🚀 Interviewer Takeaway**
🔥 **If the dataset is small**, use **Cosine Similarity or KNN**.  
🔥 **If the dataset is structured**, use **Naïve Bayes or SVM**.  
🔥 **If deep learning is required**, use **BERT-based models**.  

Would you like **more implementation details or project suggestions?** 😊
