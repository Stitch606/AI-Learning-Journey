# 🚀 Naïve Bayes SMS Classifier From Scratch (98% Accuracy)

An optimized, pure-mathematics implementation of the **Naïve Bayes** algorithm for SMS spam detection. This model was engineered completely from scratch using only **Python, NumPy, and Pandas**—strictly avoiding any high-level machine learning libraries like `scikit-learn`. 

The model successfully handles the raw probability matrix operations and outputs high-accuracy classifications on completely unseen data.

---

## 🧠 Key Features & Architectural Highlights

* **Pure NumPy Core:** Leverages accelerated element-wise matrix multiplication and vectorized broadcasting for extreme efficiency in execution.
* **Underflow Prevention:** Uses Log-Transformation (`np.log`) to convert crushing decimal probabilities into clean, manageable negative log-scales, effectively crushing the underflow floating-point memory trap.
* **Smart Dimensional Aggregation:** Utilizes rapid 1D row-compression via axis summation (`np.sum(..., axis=1)`) to compute full-text predictions dynamically in a single pass.
* **Laplace Smoothing:** Implemented custom smoothing $(+1)$ to secure the vocabulary and completely eliminate the zero-probability (`-inf`) mathematical breakdown.
* **No Overfitting/Underfitting:** Achieved a balanced, optimal sweet spot with **98% test accuracy** on 100 completely new, unseen messages after training on 5,000 samples (9,594 unique vocabulary tokens).

---

## 📊 Performance & Confusion Matrix Metric

The model was tested on a highly unbalanced slice of the dataset and showed stellar predictive precision without turning "blind":

* **Actual Dataset Slice:** 83 Ham (Normal) | 17 Spam
* **Model Predictions:** 81 Ham (Normal) | 18 Spam
* **Overall Accuracy:** ~98% on unseen testing environment!

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3 🐍
* **Data Crunching:** Pandas 🐼
* **Matrix Operations:** NumPy 🔢
* **Environment:** Google Colab Notebook 💻

---

## 🚀 How the Core Math Works (The Vectorized Code)

Instead of slow, traditional `for` loops to iterate through words, the classification boundary is computed instantly via vectorized log-likelihood matrix dotting:

```python
# The pure matrix magic that crushes text classification in one line
spam_predictions = np.sum(my_test * np.log(w_spam), axis=1) + np.log(p_spam)
ham_predictions = np.sum(my_test * np.log(w_ham), axis=1) + np.log(p_ham)

# The final crisp operational decision boundary
final_predictions = np.where(spam_predictions > ham_predictions, 1, 0)
