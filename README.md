# README

## IRIS Plant Classification using Support Vector Machine (SVM)

This project applies the Support Vector Machine (SVM) algorithm to classify the famous **IRIS dataset**, a well-known dataset in the field of machine learning. The dataset contains measurements of iris flowers (sepal length, sepal width, petal length, petal width) and aims to classify them into three species: Setosa, Versicolour, and Virginica.

### Project Workflow

1. **Data Loading**:
   - The IRIS dataset is loaded from `sklearn.datasets`.
   - `x` (features) contains the flower measurements, and `y` (target) contains the species labels.

2. **Train-Test Split**:
   - The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.
   - Stratified sampling ensures an even distribution of classes in both the training and testing sets.

3. **SVM Algorithm**:
   - The **SVM (Support Vector Classifier)** algorithm is implemented using different kernels: `rbf`, `linear`, `poly`, and `sigmoid`.
   - Performance is evaluated using a **confusion matrix** for each kernel, allowing comparison of their classification accuracy.

4. **Kernels Used**:
   - **Radial Basis Function (RBF) Kernel**:
     - Two variations are explored:
       - **RBF with Gamma = 1**: `SVC(kernel='rbf', gamma=1.0)`
       - **RBF with Gamma = 10**: `SVC(kernel='rbf', gamma=10)`
   - **Linear Kernel**: `SVC(kernel='linear')`
   - **Polynomial Kernel**: `SVC(kernel='poly')`
   - **Sigmoid Kernel**: `SVC(kernel='sigmoid')`

### Confusion Matrices

- Confusion matrices are computed for each SVM model and kernel to assess the classification performance on the test data.
  
  - **cm_rbf01**: Confusion matrix for RBF kernel with `gamma=1.0`
  - **cm_rbf10**: Confusion matrix for RBF kernel with `gamma=10`
  - **cm_linear**: Confusion matrix for Linear kernel
  - **cm_poly**: Confusion matrix for Polynomial kernel
  - **cm_sigmoid**: Confusion matrix for Sigmoid kernel

### Requirements

- Python 3.x
- Libraries:
  - `sklearn`
  - `numpy` (if needed)

Install the required libraries using:

```bash
pip install scikit-learn numpy
```

### How to Run

1. Clone this repository or download the script.
2. Run the script using Python:
   ```bash
   python iris_svm_classification.py
   ```

### Dataset

The IRIS dataset consists of 150 samples and 4 features (sepal length, sepal width, petal length, petal width). The dataset is already included in the `sklearn.datasets` module, so no external dataset is needed.

- **Classes**:
  - Setosa (Class 0)
  - Versicolour (Class 1)
  - Virginica (Class 2)

### Conclusion

This project demonstrates the use of different SVM kernels for multi-class classification of the IRIS dataset. By comparing the confusion matrices, you can evaluate which kernel performs best in terms of accuracy and misclassification.

### Author

This project was implemented as part of a learning exercise in Support Vector Machines and kernel-based methods for classification.
