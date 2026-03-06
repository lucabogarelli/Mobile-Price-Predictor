# Mobile Price Range Classification

## Context
Bob has started his own mobile phone company and needs a data-driven way to estimate the price range of his devices based on hardware features (e.g., RAM, Internal Memory, Battery Power). This project uses Machine Learning to classify mobile phones into specific price ranges, providing a competitive edge in the market.

## Dataset
The project uses the `mobileprices.csv` dataset, consisting of 2,000 samples and 21 features. The original target variable, `price_range`, contains 4 categories (0 to 3), which are mapped into a binary classification problem for simplicity:
* **0**: Low price range
* **1**: High price range

## Methodology
The code explores multiple preprocessing, sampling, dimensionality reduction, and classification techniques to find the optimal pipeline. The search space includes:
* **Transformers**: `StandardScaler`, `Normalizer`
* **Samplers** (handling data distribution): `SMOTE`, `RandomOverSampler`
* **Dimensionality Reduction**: `PCA`, `LinearDiscriminantAnalysis`, `SequentialFeatureSelector`
* **Classifiers**: `Perceptron`, `LogisticRegression`, `KNeighborsClassifier`, `RandomForestClassifier`

Using `imbalanced-learn` pipelines and `RandomizedSearchCV`, the script generates a list of candidates, selects the best performing architecture based on F1-score, and performs hyperparameter tuning on the final choice.

## Best Model & Results
The winning pipeline configuration is:
1. **Transformer**: `StandardScaler`
2. **Sampler**: `RandomOverSampler`
3. **Dimensionality Reduction**: `None`
4. **Classifier**: `LogisticRegression` (Tuned parameters: L1 penalty, balanced class weights, specific `C` value).

**Test Set Performance:**
* **F1-Score**: ~0.987
* **ROC AUC Score**: ~0.987

The project includes visualizations for the **Learning Curve** (to check for bias/variance trade-offs) and the **Validation Curve** (to evaluate the impact of the hyperparameter `C` on the Logistic Regression model).

## Requirements
* `pandas`, `numpy`
* `scikit-learn`
* `imbalanced-learn`
* `mlxtend`
* `scipy`
* `matplotlib`, `missingno`
