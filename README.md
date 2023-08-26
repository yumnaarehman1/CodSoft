# CodSoft Internship 
# Task : 1 
# Titanic Survival Prediction

Data Loading: The code starts by loading the training dataset from a CSV file named "train.csv" into a pandas DataFrame named titan_data.

Importing Libraries: Necessary libraries are imported, including pandas, seaborn, matplotlib.pyplot, and various modules from sklearn for data preprocessing and machine learning tasks.

Correlation Heatmap: A heatmap of the correlation matrix of the titan_data DataFrame is created using sns.heatmap(). This provides insight into the relationships between different numerical features in the dataset.

Stratified Train-Test Split: The dataset is split into training and testing sets using a stratified shuffle split, ensuring that the proportion of the target variable (Survived) is maintained in both sets.

Histograms: Two histograms are plotted side by side using plt.subplot(). These histograms show the distribution of Survived and Pclass features in both the training and testing sets.

Dataset Info: Information about the strat_test_set DataFrame is printed, displaying the data types and non-null counts of each feature.

Handling Missing Values: The code checks for missing values in the titan_data DataFrame using .isnull().sum(). This shows the count of missing values for each feature, sorted in descending order.

Feature Encoding: One-hot encoding is performed on categorical features (Embarked and Sex) using custom transformers (FeatureEncoder and FeatureDropper). The encoded features are added to the DataFrame.

Data Pipeline Setup: A data preprocessing pipeline is created using Pipeline. It sequentially applies an age imputer (AgeImputer), feature encoding (FeatureEncoder), and feature dropping (FeatureDropper) steps.

Pipeline Transformation: The preprocessing pipeline is applied to the strat_train_set, transforming it to the preprocessed strat_train_set.

Dataset Info (Preprocessed): Information about the preprocessed strat_train_set is printed, displaying the data types and non-null counts of each feature after preprocessing.

Standard Scaling: Features are standardized using StandardScaler to have zero mean and unit variance. The target variable y is also converted to a NumPy array.

Model Training (GridSearchCV): A RandomForestClassifier is initialized, and hyperparameters are set up for grid search. GridSearchCV is used to search for the best hyperparameters based on cross-validated accuracy. The best estimator is stored in final_clf.

Pipeline Transformation (Test Set): Similar preprocessing and transformation are applied to the strat_test_set.

Standard Scaling (Test Set): The test set features are standardized using the previously fitted StandardScaler.

Model Evaluation: The final_clf model's accuracy is evaluated on the standardized test set using the score() method.

Full Dataset Transformation: The same preprocessing pipeline is applied to the entire titan_data (training + testing) to prepare it for the final model training.

Standard Scaling (Full Dataset): Features and target are standardized for the entire dataset.

Model Training (Full Dataset): Similar to step 13, a RandomForestClassifier is trained on the fully preprocessed dataset using GridSearchCV.

Final Model Selection: The best estimator from the grid search is stored in prod_final_clf.

Loading Test Data: The test dataset is loaded from "test.csv" into the titanic_test_data DataFrame.

Pipeline Transformation (Test Data): The same preprocessing pipeline is applied to the test data.

Handling Missing Values (Test Data): Forward fill (method="ffill") is used to handle any missing values in the test data.

Standard Scaling (Test Data): The test data features are standardized using the previously fitted StandardScaler.

Predictions: The prod_final_clf model predicts survival values on the preprocessed test data.

Creating Prediction DataFrame: A DataFrame is created to store the passenger IDs and corresponding survival predictions.

Saving Predictions: The prediction DataFrame is saved to a CSV file named "predictions.csv".
