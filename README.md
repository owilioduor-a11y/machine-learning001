# Machine Learning Project — Step by Step

**Module 1 · Scikit-Learn: Linear Classification on the Iris Dataset**

A hands-on, notebook-driven machine learning learning series. Each module is a
self-contained Jupyter notebook that walks through one complete ML workflow —
from importing libraries and loading data all the way to evaluating and
cross-validating a trained model.

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## 📖 Overview

Module 1 is a complete, beginner-friendly introduction to **supervised
classification** with scikit-learn. Using the classic **Iris** dataset, it
builds up a linear classifier (`SGDClassifier`) step by step and measures its
performance honestly with both a held-out test set and 5-fold cross-validation.

## 📓 The Notebook

| Item | Detail |
| --- | --- |
| File | [`machine_learning001.ipynb`](machine_learning001.ipynb) |
| Dataset | Iris (`sklearn.datasets.load_iris`) — 150 samples, 4 features, 3 classes |
| Features used | `sepal length`, `sepal width` (first two attributes, for 2-D visualisation) |
| Model | `SGDClassifier` (linear), wrapped in a `Pipeline` with `StandardScaler` |
| Test split | 25% hold-out (`random_state=33`) → 112 train / 38 test |

## 🔍 What the notebook covers

1. **Setup** — import `IPython`, `scikit-learn`, `pandas`, `numpy` and `matplotlib`, and print their versions.
2. **Load data** — `datasets.load_iris()` into `x_iris` (150 × 4) and `y_iris` (150,).
3. **Split** — `train_test_split(test_size=0.25, random_state=33)`.
4. **Standardise** — `StandardScaler` fitted **on the training set only**, then applied to both sets.
5. **Visualise** — scatter plot of the standardised training data, coloured by class.
6. **Train** — `SGDClassifier` linear model; plots the "one class versus the rest" decision boundaries using `clf.coef_` and `clf.intercept_`.
7. **Predict** — inspect a single prediction with `clf.predict(scaler.transform([[4.7, 3.1]]))`.
8. **Evaluate** — accuracy on train and test, plus `classification_report` and `confusion_matrix`.
9. **Cross-validate** — 5-fold `KFold` (shuffled, `random_state=33`) over a `Pipeline`, then report mean ± standard error.

## 📊 Results

**Accuracy**

| Set | Accuracy |
| --- | --- |
| Training set (112 samples) | 0.8125 |
| Test set (38 samples) | 0.6842 |

**Classification report (test set)**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 | 8 |
| versicolor | 0.43 | 0.27 | 0.33 | 11 |
| virginica | 0.65 | 0.79 | 0.71 | 19 |
| **accuracy** | | | **0.68** | 38 |

> `setosa` separates perfectly, while `versicolor` and `virginica` overlap —
> expected, since only the first two features are used.

**Confusion matrix (test set)**

```
[[ 8  0  0]     # setosa     -> 8 correct
 [ 0  3  8]     # versicolor -> 3 correct, 8 predicted virginica
 [ 0  4 15]]    # virginica  -> 15 correct, 4 predicted versicolor
```

**5-fold cross-validation**

```
fold scores : [0.667, 0.800, 0.767, 0.867, 0.867]
mean ± SEM  : 0.793 (±0.037)
```

## 📂 Project Structure

```
machine learning001/
├── machine_learning001.ipynb   # Module 1 notebook
├── requirements.txt            # pinned dependencies
├── .gitattributes              # line-ending / diff normalisation
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13 (the notebook was last executed on **3.13.9**)
- `pip`

### Installation

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
cd <YOUR-REPO>

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

Then open `machine_learning001.ipynb` and **Run All** cells. No data download is
required — the Iris dataset ships with scikit-learn.

## 🧠 Key concepts covered

- Train/test splitting and using `random_state` for reproducibility
- Feature standardisation, and why the scaler is fitted on the training set only
- Linear classification: decision functions and decision boundaries
- Accuracy, precision, recall, F1-score and the confusion matrix
- Why cross-validation gives a more reliable estimate than a single split
- Reproducible pipelines with `sklearn.pipeline.Pipeline`

## 🗺️ Roadmap

- [x] **Module 1** — Linear classification with scikit-learn (Iris dataset)
- [ ] **Module 2** — coming soon
- [ ] Model selection & hyper-parameter tuning
- [ ] Tree-based ensembles
- [ ] Unsupervised learning

## 📄 License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

## 👤 Author

**Peter Owili**

## 🙏 Acknowledgements

- The **Iris** dataset (R. A. Fisher), bundled with scikit-learn.
- The scikit-learn, NumPy, pandas and Matplotlib documentation and communities.

---

⭐ If you find this series useful, consider giving the repository a star.
