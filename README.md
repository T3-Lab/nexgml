[![PyPI version](https://badge.fury.io/py/nexgml.svg)](https://pypi.org/project/nexgml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/T3-Lab/nexgml/blob/main/LICENSE) 

# NexGML (Next Generation Machine Learning)

NexGML (Next Generation Machine Learning) is a custom Machine Learning utility package built for educational and research purposes, emphasizing **code transparency** and **high performance** with efficient python implementation.

## Key Features & Technology Stack
  * **Modular Helpers:** Separates complex logic into focused helper modules (`ForLinear`, `ForTree`, `Indexing`, `Metrics`, `Guardians`) for easy customization.
  * **Sparse Data Ready:** Full support for `scipy.sparse` matrices (CSR/CSC) for memory efficiency.

## Installation
```bash
pip install nexgml
```

## 1\. Classifiers (The Models)

The primary model is the **Gradient Supported Intense Classifier (GSIC)**.

```python
from nexgml.gradient_supported import IntenseClassifier
import numpy as np

# Load data X, y...

model = IntenseClassifier(
    optimizer='adamw', 
    lr_scheduler='plateau', 
    batch_size=32, 
    penalty='elasticnet'
)
model.fit(X_train, y_train)

print(f"Final Training Loss: {model.loss_history[-1]:.6f}")
```

## 2\. Regressors (The Models)

The primary model is the **Gradient Supported Intense Regressor (GSIR)**.

```python
from nexgml.gradient_supported import IntenseRegressor
import numpy as np

# Load data X, y...

model = IntenseClassifier(
    optimizer='adamw', 
    lr_scheduler='plateau', 
    batch_size=32, 
    penalty='elasticnet'
)
model.fit(X_train, y_train)

print(f"Final Training Loss: {model.loss_history[-1]:.6f}")
```

## 3\. Helper Modules (Performance Backbone)

These modules contain the high-speed math used internally.

| Module | Purpose | Example Usage |
| :--- | :--- | :--- |
| `nexgml.amo.forlinear` | **Linear Criteria.** Activation/Loss functions (Softmax, CCE, RMSE). | `forlinear.softmax(logits)` |
| `nexgml.amo.fortree` | **Tree Criteria.** Impurity measures (Gini, Entropy, Friedman MSE). | `fortree.gini_impurity(labels)` |
| `nexgml.indexing` | **Data Utilities.** One-hot encoding, smart feature slicing (`standard_indexing`). | `indexing.standard_indexing(n_features, 'sqrt')` |
| `nexgml.metrics` | **Model Metrics.** Regressor and classifier models metrics computation (R^2, F1, Accuracy Score) | `accuracy_score(y_true, pred)` |
| `nexgml.guardians` | **Numerical stability**. Value clipping, invalid value detecting (safe_array, hasinf, hasnan) | `safe_array(array)` |

## 📝 Documentation & Exploration

This repository is dedicated to experimentation, learning, and personal research, primarily in the following fields:

- 🤖 Artificial Intelligence and Machine Learning
- 💻 Python development and performance optimization
- 📖 Technical documentation and concept notes
- 🥼 Designing and developing concepts, theories and hypotheses. Before finally forming its implementation

> ⚠️ This project is intended for exploration and learning purposes only.

> If you find this repo helpful or interesting, feel free to fork, star, or open a pull request.  
This is a learning space—no pressure, just passion! 😄