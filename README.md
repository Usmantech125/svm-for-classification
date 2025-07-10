# 🔍 SVM Binary Classification - Breast Cancer Prediction

This project implements a **Support Vector Machine (SVM)** model for binary classification using the **Breast Cancer Wisconsin** dataset. The task was completed as part of the **CodVeda Internship**.

## 📌 Objectives

- Train an SVM model on a labeled dataset
- Use two different kernels:
  - Linear
  - RBF (Radial Basis Function)
- Compare performance metrics
- Visualize the decision boundaries
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - AUC (Area Under the ROC Curve)

---

## 📊 Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features used for visualization:**  
  - `radius_mean`  
  - `perimeter_mean`
- **Target variable (`diagnosis`):**
  - `M` = Malignant → 1  
  - `B` = Benign → 0

---

## ⚙️ Tools & Libraries

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 🧠 Model Training

We trained two SVM models with the following kernels:

- `SVC(kernel='linear')`
- `SVC(kernel='rbf')`

The models were trained using only 2 features to make visualization of the decision boundary possible.

---

## 📈 Results

### ✅ Linear SVM Performance:
- **Accuracy:** 92.98%
- **Precision:** 97.30%
- **Recall:** 83.72%
- **AUC:** 98.03%

### ✅ RBF SVM Performance:
- **Accuracy:** 92.11%
- **Precision:** 97.22%
- **Recall:** 81.39%
- **AUC:** 98.00%

---

## 🖼️ Decision Boundary Visualization

Both SVM models' decision boundaries were plotted using a mesh grid over the 2D feature space. The plots clearly show how the two kernels separate malignant vs benign cases.

---

## 🧪 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python svm_classifier.py
