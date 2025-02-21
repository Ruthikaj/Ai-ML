
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

These questions cover core Machine Learning concepts, particularly Decision Trees, Random Forest, K-Nearest Neighbors (KNN), and classification techniques. Here's a structured breakdown of the answers:

---

### **Q1: What is a Decision Tree? How to split? How does a Decision Tree work?**  
- **Decision Tree** is a supervised learning algorithm used for classification and regression. It splits the dataset into branches based on feature values.  
- **Splitting**: A node is split using metrics like  
  - **Gini Impurity** (default in scikit-learn)  
  - **Entropy (Information Gain)**  
  - **Chi-square** (for categorical data)  
  - **Reduction in Variance** (for regression)  
- **How it works?**  
  1. Choose the best feature to split using Gini/Entropy.  
  2. Recursively split the dataset until stopping conditions (like max depth, min samples, etc.) are met.  
  3. The leaf nodes represent the final class or predicted value.

---

### **Q2: What does each node contain in a Decision Tree?**  
Each node consists of:  
- **Feature** used for the split.  
- **Threshold** value for numerical splits.  
- **Gini impurity or entropy** to measure homogeneity.  
- **Number of samples** at that node.  
- **Distribution of target classes** (for classification).  
- **Predicted output** (majority class for classification or mean value for regression).  

---

### **Q3: What is Entropy and Gini Index? How does it help?**  
- **Entropy**: Measures the randomness in data. Lower entropy means purer splits.  
  \[
  H(X) = -\sum p_i \log_2 p_i
  \]  
  Higher entropy means more disorder, requiring better splits.  
- **Gini Index**: Measures impurity of a node.  
  \[
  G = 1 - \sum p_i^2
  \]  
  - If Gini = 0, all instances belong to a single class (pure node).  
  - Lower Gini means better splits.  
**Use case**:  
- Entropy is slower but gives fine-grained splits.  
- Gini is computationally faster and works well in most cases.

---

### **Q4: What is Random Forest? What is “Random” in Random Forest? How to calculate OOB Error?**  
- **Random Forest** is an ensemble of multiple decision trees. It combines weak learners to improve accuracy and reduce overfitting.  
- **Randomness** comes from:  
  - **Bootstrap Sampling** (each tree gets a random subset of data).  
  - **Feature Selection** (each split considers only a random subset of features).  
- **Out-of-Bag (OOB) Error**:  
  - Each tree is trained on a bootstrap sample (about 63% of data).  
  - Remaining 37% (out-of-bag samples) are used to test the tree’s performance.  
  - OOB Error is the average error across all trees on their respective OOB samples.  
  - Helps estimate generalization error without needing a validation set.

---

### **Q5: How does Random Forest work?**  
1. **Bootstrap Sampling**: Multiple samples are drawn with replacement.  
2. **Feature Selection**: Each tree selects random subsets of features.  
3. **Tree Building**: Each tree is trained on its subset using the Decision Tree algorithm.  
4. **Aggregation**:  
   - For classification → majority voting.  
   - For regression → average prediction.  
5. **Final Prediction**: Combines results from all trees to improve accuracy.

---

### **Q6: Explain the entire process from data collection to final prediction.**  
1. **Data Collection**: Gather structured/unstructured data from sources.  
2. **Data Preprocessing**: Handle missing values, scaling, encoding categorical features.  
3. **Feature Selection & Engineering**: Identify relevant features, create new features if needed.  
4. **Train-Test Split**: Divide data into training and test sets (e.g., 80-20 split).  
5. **Model Selection**: Choose an appropriate algorithm (Decision Tree, Random Forest, KNN, etc.).  
6. **Hyperparameter Tuning**: Optimize parameters using Grid Search, Random Search, or Bayesian Optimization.  
7. **Model Training**: Fit the selected model to the training data.  
8. **Model Evaluation**: Use metrics like Accuracy, Precision, Recall, F1-score, AUC-ROC, RMSE, etc.  
9. **Deployment**: Deploy the model using Flask, FastAPI, or cloud-based services.  
10. **Monitoring & Maintenance**: Continuously track model performance and retrain if necessary.

---

### **Q7: How does KNN work? Which distance metric to use for categorical data?**  
- **KNN (K-Nearest Neighbors)** is a lazy learning algorithm where a new instance is classified based on the majority class of its `k` nearest neighbors.  
- **Distance Metrics for Categorical Data**:  
  - **Hamming Distance** (for binary/categorical data)  
  - **Jaccard Similarity** (for text data)  
  - **Chi-square distance** (for ordinal categorical features)  

---

### **Q8: You have 10 documents. Each topic has been tagged. When a new document arrives, how to tag it?**  
- **Approach**: Text Classification using NLP  
  1. **Convert text to numerical representation**  
     - TF-IDF  
     - Word Embeddings (Word2Vec, GloVe, BERT)  
  2. **Train a classifier**  
     - Naïve Bayes (for small datasets)  
     - Logistic Regression  
     - SVM  
     - Random Forest  
     - Deep Learning (LSTM, Transformer)  
  3. **Predict label for new document** using the trained classifier.  
  4. **Alternative**: Use **KNN (TF-IDF + Cosine Similarity)** to assign the closest topic.  

---

### **Conclusion**
The candidate should be strong in:
- **ML Algorithms (Decision Trees, Random Forest, KNN)**
- **Coding Skills (Python, scikit-learn, pandas, NumPy)**
- **Feature Engineering, Data Preprocessing**
- **Evaluation Metrics & Model Optimization**
- **Text Classification & NLP Techniques**



