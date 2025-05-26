# Stroke Data Clustering Analysis

## Overview
This Jupyter Notebook (`Milestone_2 (1).ipynb`) performs clustering analysis on a stroke dataset to identify natural groupings of patients based on their characteristics and assess stroke risk patterns. The notebook uses unsupervised learning techniques, specifically K-means and Hierarchical Clustering, and compares the results with supervised learning classifier predictions. The goal is to uncover insights into patient risk profiles that may complement supervised learning approaches.

## Dataset
The dataset used is `healthcare-dataset-stroke-data (2).csv`, which contains patient information related to stroke risk factors. Key columns include:
- **id**: Unique patient identifier
- **gender**: Patient's gender
- **age**: Patient's age
- **hypertension**: Binary indicator (0 or 1) for hypertension
- **heart_disease**: Binary indicator (0 or 1) for heart disease
- **ever_married**: Marital status (Yes/No)
- **work_type**: Type of employment
- **Residence_type**: Urban or Rural residence
- **avg_glucose_level**: Average glucose level
- **bmi**: Body Mass Index
- **smoking_status**: Smoking habits
- **stroke**: Target variable (0 for no stroke, 1 for stroke)

## Prerequisites
To run this notebook, ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `sklearn` (scikit-learn)
- `optuna`
- `seaborn`
- `matplotlib`
- `scipy`

You can install the required libraries using the command:
```bash
pip install pandas numpy scikit-learn optuna seaborn matplotlib scipy
```

The notebook also installs `optuna` explicitly via `!pip install optuna` in the first cell.

## Notebook Structure
The notebook is organized into several sections, each marked by markdown headers or code cells. Below is a summary of the key sections:

### 1. **Library Installation**
- The first cell installs the `optuna` library for hyperparameter optimization (though its usage is not fully shown in the provided notebook snippet).
- Output confirms the installation of `optuna` and its dependencies.

### 2. **Importing Modules**
- Imports essential Python libraries for data manipulation (`pandas`, `numpy`), preprocessing (`StandardScaler`, `OneHotEncoder`), dimensionality reduction (`PCA`, `TSNE`, `LinearDiscriminantAnalysis`), machine learning models (`GaussianNB`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `KMeans`, `AgglomerativeClustering`), hyperparameter tuning (`GridSearchCV`, `optuna`), and visualization (`seaborn`, `matplotlib`).
- Suppresses warnings using `warnings.filterwarnings('ignore')` to keep the output clean.

### 3. **Load Dataset**
- Loads the stroke dataset (`healthcare-dataset-stroke-data (2).csv`) into a pandas DataFrame (`stroke_df`).
- Displays the first five rows of the dataset to provide an overview of its structure and content.

### 4. **Clustering Analysis Summary**
- Summarizes the results of clustering analysis performed on the dataset (though the actual clustering code is not fully shown in the provided snippet).
- Key points include:
  - **Hierarchical Clustering**: A dendrogram visualizes patient relationships, identifying clusters with varying stroke risk levels.
  - **K-means Clustering**: Uses the elbow method to determine the optimal number of clusters (k=2), revealing distinct patient characteristic patterns and stroke risk levels.
  - **Comparison with Classifiers**: Clustering results show partial alignment with classifier predictions, identifying high-risk groups and providing complementary insights.
  - **Key Insights**: Identifies natural patient groupings, highlights clusters with different stroke risks, and reveals patterns not easily detected by supervised learning.
- Reports an agreement of 62.27% between K-means clustering and classifier predictions.

## Key Features
- **Data Preprocessing**: Likely includes handling missing values (e.g., `bmi` has NaN values), encoding categorical variables (e.g., `gender`, `work_type`, `smoking_status`), and scaling numerical features (e.g., `age`, `avg_glucose_level`, `bmi`) using tools like `StandardScaler` and `OneHotEncoder`.
- **Dimensionality Reduction**: Uses techniques like PCA, TSNE, or LDA to reduce feature dimensions for clustering and visualization.
- **Clustering**: Implements K-means and Hierarchical Clustering to group patients based on features, with the elbow method for K-means optimization.
- **Visualization**: Likely includes plots such as dendrograms (for hierarchical clustering) and scatter plots (for K-means or TSNE results) using `seaborn` and `matplotlib`.
- **Model Evaluation**: Compares clustering results with supervised classifiers (e.g., Naive Bayes, SVM, KNN, Decision Trees) using metrics like accuracy, precision, recall, and F1-score.

## Expected Outputs
- **Data Overview**: Displays the first few rows of the dataset.
- **Clustering Results**:
  - Dendrogram for hierarchical clustering.
  - Elbow plot for determining the optimal number of clusters in K-means.
  - Cluster assignments and their stroke risk profiles.
- **Classifier Comparison**: Metrics showing alignment between clustering and classifier predictions (e.g., 62.27% agreement).
- **Visualizations**: Plots illustrating patient groupings and feature distributions.

## How to Run
1. **Set Up Environment**:
   - Ensure all required libraries are installed (see Prerequisites).
   - Place the `healthcare-dataset-stroke-data (2).csv` file in the same directory as the notebook or update the file path in the `pd.read_csv()` call.
2. **Run the Notebook**:
   - Open the notebook in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to install dependencies, load data, and perform clustering analysis.
3. **Interpret Results**:
   - Review the clustering summary for insights into patient groupings.
   - Check visualizations for patterns in stroke risk.
   - Compare clustering results with classifier predictions to understand complementary insights.

## Limitations
- The dataset may have missing values (e.g., `bmi`), which require preprocessing.
- The notebook snippet does not show the full clustering code, so some implementation details (e.g., feature selection, preprocessing steps) are assumed.
- The dataset is imbalanced (stroke cases are rare), which may affect clustering and classifier performance.
- The 62.27% agreement between K-means and classifiers suggests moderate alignment, indicating potential for further refinement.

## Future Improvements
- Handle missing values explicitly (e.g., impute `bmi` using mean/median or advanced methods).
- Experiment with additional clustering algorithms (e.g., DBSCAN, Gaussian Mixture Models).
- Use advanced feature selection techniques to improve clustering quality.
- Address class imbalance using techniques like SMOTE before clustering or classification.
- Include more detailed visualizations, such as feature importance plots or cluster-specific stroke risk profiles.

## License
This project is for educational purposes and uses a publicly available dataset. Ensure compliance with any dataset-specific licensing terms.

## Contact
For questions or contributions, please contact the notebook author or repository maintainer.
