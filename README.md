# Machine_learning_examples
Examples of ML codes and blocks found online.

## CloudAcademy
This folder contains two files, with examples of regression and classification algorithms:
 1. **01-Regression-stock-full.ipynb**
    - ```pandas.data.DataReader```: read financial data
    - Missing values imputing (*IterativeImputer*)
    - Features scaling (*StandardScaler*)
    - Linear regression (coefficient visualization, $R^2$ score, MSE score)
    - Ridge regression
    - Model tuning: Grid search cross-validation (*GridSearchCV*)
 2. **02-classification-stroke.ipynb**
    - Categorical features encoding (*OneHotEncoder*)
    - Missing values imputing (*IterativeImputer*)
    - Dealing with imbalanced data: ```umblearn.over_sampling``` (*SMOTE*), to create new, synthetic data
    - Features scaling (*StandardScaler*)
    - Logistic regression
    - Grid search cross-validation (*GridSearchCV*)
    - Pipeline
    - Confusion matrix
