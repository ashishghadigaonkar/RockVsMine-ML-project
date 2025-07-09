Sonar Data Classification
This project aims to classify sonar signals as either a "Rock" or a "Mine" using various machine learning models. The dataset consists of 60 numerical features representing the sonar signal and a target variable indicating whether the object is a rock or a mine.

Project Steps:
Import Libraries: Import necessary libraries for data manipulation, machine learning, and visualization.
Load and Prepare Data: Load the sonar dataset, examine its dimensions and first few rows, separate features (X) and target (y), and convert the target variable to a binary format (0 for Rock, 1 for Mine).
Split and Scale Data: Split the data into training and testing sets and standardize the features using StandardScaler.
KNN Model:
Train a K-Nearest Neighbors (KNN) model.
Evaluate the model's performance using accuracy, precision, recall, and a confusion matrix.
Implement hyperparameter tuning with GridSearchCV to find the best KNN parameters.
Utilize SMOTE for handling class imbalance.
Apply a custom threshold to the prediction probabilities to potentially improve recall for the 'Mine' class.
Visualize the confusion matrix and probability distributions.
Explore methods to improve KNN accuracy, including testing different K values and using Robust Scaling and a hybrid distance metric.
Random Forest Model:
Train a Random Forest classifier.
Evaluate the model's performance using accuracy, precision, recall, and a confusion matrix.
Implement hyperparameter tuning with GridSearchCV, considering class weights and different max_features strategies.
Apply a custom threshold to the prediction probabilities.
Visualize the confusion matrix.
SVM Model:
Train a Support Vector Machine (SVM) model with an RBF kernel.
Evaluate the model's performance using accuracy, precision, recall, and a confusion matrix.
Visualize the confusion matrix.
Model Comparison:
Create a table comparing the performance of the KNN, Random Forest, and SVM models based on accuracy, precision, and recall.
Generate bar plots to visualize the model comparison.
Generate ROC curves and Precision-Recall curves to compare model performance visually.
Visualize feature importance for the Random Forest model.
Generate a Learning Curve for the KNN model to assess its performance with varying training data size.
Generate a clustermap of the feature correlation matrix.
Save Models and Data: Save the trained models and the test data using joblib for future use.
Dataset:
The dataset used in this project is sonar data (1).csv. It contains 208 instances and 61 attributes. The first 60 attributes are the sonar signal features, and the last attribute is the target variable ('R' for Rock, 'M' for Mine).

Dependencies:
pandas
numpy
sklearn
seaborn
matplotlib
imblearn
plotly
joblib
How to Run the Code:
Ensure you have the required libraries installed (pip install pandas numpy scikit-learn seaborn matplotlib imblearn plotly joblib).
Make sure the sonar data (1).csv file is in the same directory as the notebook or provide the correct path.
Run the code cells sequentially in the provided Jupyter Notebook or Python script.
Results:
The notebook provides the performance metrics (accuracy, precision, recall) and confusion matrices for each model. It also includes visualizations for comparing the models, exploring data characteristics, and analyzing model behavior. The best performing model and its optimal hyperparameters are identified through hyperparameter tuning.

