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

## GreatLearning
This folder contains a simple example of a **deep learning** algorithm for **classification** using *keras*:
  - ```sklearn.datasets.load_iris```: download iris data
  - ```seaborn.pairplot```: make matrix of distributions and scatter plots of all features
  - heatmap
  - ```sklearn.utils.shuffle```: shuffle datasets when targets are grouped (could bias the classifier)
  - Data scaling (*StandardScaler*)
  - Feature encoding (*LabelBinarizer* from sklearn, similar to OneHotEncoder)
  - ```train_test_split```: to divide the data into train and test
  - Model training (*Sequential*)
  - *Confusion matrix*
  - Accuracy/Score VS Epoch
