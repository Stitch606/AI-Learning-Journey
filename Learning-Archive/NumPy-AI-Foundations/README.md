# 🚀 My NumPy Odyssey: Building AI from absolute scratch

Welcome to my journey into the core of Artificial Intelligence! This repository is a living document and a technical log of my transition into becoming an **AI and Machine Learning Engineer**. 

Here, you won't find high-level framework wrappers or five-line ready-made models. Instead, you will find the raw beauty of **Classical Machine Learning** and **Deep Learning** algorithms built entirely from scratch, utilizing the power of linear algebra and numerical computing through **NumPy**.

> *"True mastery begins when you build the foundations with your own hands."*

---

## 📌 Why from Scratch?
In a world driven by automated tools, I believe that a great engineer must understand the inner workings of the algorithms. By avoiding high-level libraries (like Scikit-Learn or PyTorch) in this phase, I gain:
- Absolute ownership of the code and math.
- Deep intuition of optimization techniques, gradients, and numerical stability.
- The ability to debug complex architectural issues from the ground up.

---

## 🗺️ Roadmap & Implemented Algorithms

### 🧠 1. Classical Machine Learning
Focusing on foundational statistical models and optimization algorithms.
- [x] **Logistic Regression:** Built using raw matrix dot products, custom learning rate schedules, and sigmoid activation.
- [x] **Numerical Stability & Scaling:** Implemented custom **Min-Max Scaling** from scratch, with strict boundary protections (`np.clip`) to prevent exponent overflow.
- [ ] **K-Nearest Neighbors (KNN)** *(In Progress)*
- [ ] **Softmax Regression:** (Multi-class classification with stable vectorization across rows).

### ⚡ 2. Deep Learning
Moving towards neural network architectures, backpropagation, and deep representations.
- [ ] **Perceptron & Multi-Layer Perceptron (MLP)**
- [ ] **Custom Backpropagation Engine:** Manual partial derivatives and chain-rule activation tracking.
- [ ] **Deep Neural Networks (DNN):** Custom layers, weights initialization, and activation functions (ReLU, Sigmoid, Tanh).

---

## 🛠️ Tech Stack & Environment
- **Core Language:** Python 3
- **Primary Library:** NumPy (Numerical Computing & Matrix Operations)
- **Data Handling:** Pandas (Strictly restricted to initial data loading and preparation)
- **Environment:** Google Colab (Cloud-based development)
