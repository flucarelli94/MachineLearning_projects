This folder contains some personal machine learning (ML) projects. Such projects are implemented in Python, using the sklearn module and its submodules

# House prices
Contains the solution of a **regression problem**: determine the price of houses, using numerical and categorical predictors</br>
Covered topics:
 * Data load and inspection (feature distributions, **Pearson correlations**, heatmap, boxplots, scatter plots, ...)
 * Handling of missing values (**IterativeImputer**)
 * Encoding of categorical features (**OrdinalEncoder**, **LabelEncoder**)
 * Handling skewed data (**skewness** calculation, log1p symmetrization)
 * Data scaling (**StandardScaler**, **RobustScaler**)
 * Model hyperparameter tuning (**GridSearchCV**)
 * Score estimation (**RMS error**)
 * Pipeline construction (**Pipeline**)
 * **Cross validation**
 * Performance visualization (true target/predictions, ratio plot)
 * Feature importance
 * Model prediction combination (weighted average)
 * ML algorithms: **Linear regression**, **Ridge regression**, **LASSO**, **kNN**, **Gradient Boosting Regressor**, **Random Forest**, **Support Vector Machine**

# Titanic
Contains the solution of a **classification problem**: determine whether a subset of test passengers survive or not the disaster, using numerical and categorical predictors.</br>
Covered topics:
 * Data load and inspection
 * Handling of **missing values**
 * Handling of categorical features (**OneHotEncoder**)
 * Data **scaling**
 * **Cross validation**
 * Score estimatation
 * ML algorithms: **Logistic regression**, **kNN**, **Naive Bayes Classifier**, **Decision Tree**, **Random Forest**

# Floods
Contains a model for a **classification problem**: determine whether in a given year there will likely be floods or not. Note: the data of this project are not provided, only the jupyter notebook is available.</br>
Covered topics:
 * Data load and inspection (heatmap, **Pearson correlation**, boxplots, histograms, scatter plots, ...)
 * Handling of missing values (**IterativeImputer**)
 * Encoding of categorical features (**LabelEncoder**)
 * Feature combination
 * Dealing with imbalanced data (**SMOTE**)
 * Handling skewed data (**skewness**, log1p symmetrization)
 * **Pipeline**
 * **Cross validation**
 * Hyperparameter tuning (**GridSearchCV**)
 * Data scaling (**StandardScaler**)
 * Performance visualization (**confusion matrix**)
 * **Overtraining** check
 * ML algorithms: **Logistic regression**, **Linear Discriminant Analysis**, **Decision Tree**, **Random forest**, **kNN**
