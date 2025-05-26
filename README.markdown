# Stroke Prediction Classification Analysis (Milestone_2 (1).ipynb)

## Overview
This Jupyter Notebook (`Milestone_2 (1).ipynb`) focuses on building and evaluating classification models to predict stroke risk using the `healthcare-dataset-stroke-data (2).csv` dataset. The primary goal is to develop supervised learning models to classify patients as having a stroke (1) or not (0) based on their characteristics. Additionally, clustering analysis (K-means and Hierarchical Clustering) is performed to explore natural patient groupings and compare them with classifier predictions for deeper insights.

## Dataset
The dataset used is `healthcare-dataset-stroke-data (2).csv`, containing patient information related to stroke risk factors. Key columns include:
- **id**: Unique patient identifier
- **gender**: Patient's gender (Male, Female)
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

Install the required libraries using:
```bash
pip install pandas numpy scikit-learn optuna seaborn matplotlib scipy
```

The notebook includes a cell to install `optuna` explicitly via `!pip install optuna`.

## Notebook Structure
The notebook is organized into sections, each marked by markdown headers or code cells. Below is a summary of the key sections:

### 1. **Library Installation**
- Installs the `optuna` library for hyperparameter optimization.
- Output confirms the installation of `optuna` and its dependencies (e.g., `alembic`, `sqlalchemy`).

### 2. **Importing Modules**
- Imports libraries for:
  - Data manipulation: `pandas`, `numpy`
  - Preprocessing: `StandardScaler`, `OneHotEncoder`
  - Dimensionality reduction: `PCA`, `TSNE`, `LinearDiscriminantAnalysis`
  - Classification models: `GaussianNB`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`
  - Hyperparameter tuning: `GridSearchCV`, `optuna`
  - Clustering: `KMeans`, `AgglomerativeClustering`
  - Visualization: `seaborn`, `matplotlib`
  - Evaluation: `accuracy_score`, `classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`
- Suppresses warnings using `warnings.filterwarnings('ignore')` for cleaner output.

### 3. **Load Dataset**
- Loads the stroke dataset into a pandas DataFrame (`stroke_df`).
- Displays the first five rows to show the dataset's structure, including features and the target variable (`stroke`).

### 4. **Classification Analysis**
- **Objective**: Build and evaluate supervised learning models to predict stroke risk.
- **Models Used**: Likely includes Naive Bayes (`GaussianNB`), Support Vector Classifier (`SVC`), K-Nearest Neighbors (`KNeighborsClassifier`), and Decision Trees (`DecisionTreeClassifier`), based on imported modules.
- **Preprocessing**: 
  - Handles missing values (e.g., `bmi` has NaN values).
  - Encodes categorical variables (e.g., `gender`, `work_type`, `smoking_status`) using `OneHotEncoder`.
  - Scales numerical features (e.g., `age`, `avg_glucose_level`, `bmi`) using `StandardScaler`.
  - May apply dimensionality reduction (e.g., `PCA`, `TSNE`, `LDA`) to improve model performance or visualization.
- **Model Training and Evaluation**:
  - Splits data into training and testing sets using `train_test_split`.
  - Optimizes model hyperparameters using `GridSearchCV` or `optuna`.
  - Evaluates models using metrics like accuracy, precision, recall, F1-score, and confusion matrices.
- **Visualization**: Likely includes plots (via `seaborn`, `matplotlib`) such as feature distributions, confusion matrices, or ROC curves to assess model performance.

### 5. **Clustering Analysis for Comparison**
- **Objective**: Perform clustering to identify natural patient groupings and compare with classifier predictions.
- **Methods**:
  - **Hierarchical Clustering**: Generates a dendrogram to visualize patient relationships, identifying clusters with varying stroke risk.
  - **K-means Clustering**: Determines the optimal number of clusters (k=2) using the elbow method, revealing distinct patient characteristic patterns.
- **Comparison**: Reports a 62.27% agreement between K-means clustering and classifier predictions, indicating partial alignment.
- **Insights**:
  - Clusters reveal high-risk groups that align with classifier predictions.
  - Identifies patterns in patient data not easily detected by supervised models.
  - Provides complementary insights to enhance understanding of stroke risk factors.

### 6. **Clustering Analysis Summary**
- Summarizes clustering results:
  - **Hierarchical Clustering**: Shows patient relationships and stroke risk variations across clusters.
  - **K-means Clustering**: Identifies two clusters with distinct stroke risk profiles.
  - **Classifier Comparison**: Highlights partial alignment (62.27%) and complementary insights.
  - **Key Insights**: Uncovers natural patient groupings, varying stroke risk levels, and patterns that enhance classification results.

## Key Features
- **Classification Focus**: Builds robust models to predict stroke risk, with hyperparameter tuning for optimal performance.
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features.
- **Model Evaluation**: Uses comprehensive metrics (accuracy, precision, recall, F1-score) and visualizations (e.g., confusion matrices).
- **Clustering Comparison**: Applies K-means and Hierarchical Clustering to validate classification results and uncover additional patterns.
- **Visualization**: Includes plots for feature analysis, model performance, and cluster visualization (e.g., dendrograms, scatter plots).

## Expected Outputs
- **Data Overview**: First five rows of the dataset.
- **Classification Results**:
  - Model performance metrics (accuracy, precision, recall, F1-score).
  - Confusion matrices and other visualizations (e.g., ROC curves).
- **Clustering Results**:
  - Dendrogram for hierarchical clustering.
  - Elbow plot for K-means cluster optimization.
  - Cluster assignments with stroke risk profiles.
- **Comparison**: 62.27% agreement between K-means and classifier predictions.
- **Visualizations**: Plots for feature distributions, model performance, and cluster patterns.

## How to Run
1. **Set Up Environment**:
   - Install required libraries (see Prerequisites).
   - Place `healthcare-dataset-stroke-data (2).csv` in the notebook's directory or update the file path in `pd.read_csv()`.
2. **Run the Notebook**:
   - Open in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially to install dependencies, load data, preprocess, train models, and perform clustering.
3. **Interpret Results**:
   - Review classification metrics to assess model performance.
   - Examine clustering results for patient groupings and stroke risk patterns.
   - Compare clustering and classification outputs to understand complementary insights.

## Limitations
- **Class Imbalance**: Stroke cases are rare, potentially affecting model performance.
- **Missing Values**: `bmi` contains NaN values, requiring imputation.
- **Partial Clustering Code**: The notebook snippet does not show full clustering implementation, so some details are assumed.
- **Moderate Agreement**: The 62.27% agreement between clustering and classifiers suggests room for improved alignment.

## Future Improvements
- **Enhance Preprocessing**: Explore advanced imputation methods for missing `bmi` values.
- **Model Expansion**: Test additional classifiers (e.g., Random Forest, XGBoost) for better performance.
- **Clustering Refinement**: Experiment with other clustering algorithms (e.g., DBSCAN) or feature selection to improve cluster quality.
- **Visualization**: Add feature importance plots or detailed cluster-specific stroke risk analysis.

