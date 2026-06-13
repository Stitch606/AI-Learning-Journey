# 🌳 Custom Decision Tree from Scratch

Welcome! In this project, I built a complete **Decision Tree algorithm** from scratch using Python. I did not use any ready-made machine learning libraries (like Scikit-Learn) for training, because my goal was to understand the actual logic, math, and mechanics behind how machine learning models work under the hood.

---

## 🚀 About the Project
This project takes patient data, finds the best splitting rules (thresholds), and builds a tree structure. It can then predict a new patient's status with high accuracy. Additionally, it includes a simple concept of automated model evaluation.

### 🔥 Key Features:
* **100% From Scratch:** The training logic and tree-splitting mechanics are built using first principles.
* **Smart Predict Function:** Uses recursion to travel through nested dictionaries until it reaches the final decision.
* **AutoML Concept:** A built-in function to quickly test and compare multiple algorithms at once, saving hours of work during freelance projects.

---

## 📦 Code Structure & Design

The project is organized cleanly into easy-to-use functions:
1. `tree(x, y)`: Builds the nested dictionary tree and extracts the splitting rules.
2. `predict(tree_dict, patient)`: Travels through the branches to predict the status for a single patient.
3. `evaluate_all_models(x, y)`: A control panel to compare accuracy across different models instantly.

---

## 💻 How to Use This Library

You can import and use this custom logic cleanly in just a few lines:

```python
from my_ml_library import predict, tree

# 1. Train the tree and build the model's brain
my_model = tree(x, y)

# 2. Predict the status of a single patient
final_answer = predict(my_model, single_patient)
