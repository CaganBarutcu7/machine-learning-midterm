# machine-learning-midterm
This project examines machine learning classification and clustering methods using two datasets from scikit-learn: Breast Cancer Wisconsin and Wine Classification.

Datasets:
Breast Cancer Wisconsin
569 samples
30 numerical features
2 classes
Source: scikit-learn load_breast_cancer()

Wine Classification
178 samples
13 numerical features
3 classes
Source: scikit-learn load_wine()

Classification algorithms:
Decision Tree
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Clustering algorithm:
K-Means

Classification models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.
K-Means results were evaluated using Adjusted Rand Index (ARI) and Silhouette Score.

Python: 3.9.12
pandas: 1.4.2
numpy: 1.21.5
matplotlib: 3.5.1
seaborn: 0.11.2
scikit-learn: 1.0.2

For Installation;
Download or clone this repository.
Install Python and JupyterLab.
Open a terminal in the project folder.
Install the required packages:
pip install -r requirements.txt
Sart JupyterLab:
jupyter lab
Open breast_cancer.ipynb and wine.ipynb.
Run the notebook cells from top to bottom.

Project Files;

breast_cancer.ipynb: Breast Cancer dataset analysis
wine.ipynb: Wine dataset analysis
report.docx: Project report
figures/: Generated graphs
classifier_runs.csv: Classification experiment results
classifier_summary.csv: Classification summary results
classifier_test.csv: Final classification test results
kmeans_runs.csv: K-Means experiment results
kmeans_summary.csv: K-Means summary results
kmeans_test.csv: Final K-Means test results
requirements.txt: Required Python packages
AI_DISCLOSURE.md: AI assistance declaration
