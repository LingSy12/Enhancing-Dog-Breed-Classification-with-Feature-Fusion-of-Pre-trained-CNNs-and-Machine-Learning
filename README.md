# Enhancing Dog Breed Classification with Feature Fusion of Pre‑trained CNNs and Traditional Machine Learning Classifiers

Master's project (MSc Computer Science, UNIMAS). This project improves dog‑breed
classification by **fusing deep features extracted from pre‑trained CNN backbones**
and feeding the combined feature vectors into **traditional machine‑learning
classifiers** (logistic regression) instead of a single end‑to‑end deep network.

## 💡 Idea

1. Pass each dog image through pre‑trained CNNs and extract their feature embeddings.
2. **Fuse** the features from multiple networks into one combined representation.
3. Train a classical classifier (logistic regression) on the fused features.

Combining complementary CNN features with a lightweight classifier keeps training
fast while reaching accuracy competitive with much heavier deep models.

## 📊 Results

| Notebook | Train/Test split | Classifier | Accuracy |
|---|---|---|---|
| `70-30 logistic regression 94.10.ipynb` | 70 / 30 | Logistic Regression | ~94.1% |
| `full-alphabet-80-20-logistic-regression-94-9.ipynb` | 80 / 20 (all breeds) | Logistic Regression | ~94.9% |

> Accuracies are reported from the experiment runs saved in each notebook.

## 🛠️ Tech

- Python · Jupyter Notebook
- Pre‑trained CNN feature extractors (transfer learning)
- scikit‑learn (logistic regression) for the fused‑feature classifier

## ▶️ How to run

1. Open the notebooks in Jupyter, Google Colab or Kaggle.
2. Point the data loader at your dog‑breed image dataset.
3. Run the cells top to bottom to extract features, fuse them, train and evaluate.

## 📁 Repository

- `70-30 logistic regression 94.10.ipynb` — feature‑fusion + logistic regression, 70/30 split
- `full-alphabet-80-20-logistic-regression-94-9.ipynb` — feature‑fusion + logistic regression, 80/20 split across all breeds
