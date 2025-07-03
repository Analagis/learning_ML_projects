## Project Summary
This repository contains implementation of linear regression models with various regularization techniques, feature engineering, and model evaluation for a real estate dataset (from Kaggle). The main focus is on understanding the behavior of linear models, overfitting detection, regularization, and performance comparison.

📌 Contents
1. Data Preprocessing
Imported train/test datasets.
Preprocessed Interest Level categorical feature.
Cleaned and expanded the Features column into 20 binary features (e.g., 'Elevator', 'CatsAllowed', etc.).
Final feature set includes: bathrooms, bedrooms, interest_level, and 20 new binary features.
2. Model Implementation
Custom Python classes implemented for:
Linear Regression (with analytical solution, SGD, and GD)
Ridge (L2) , Lasso (L1) , and ElasticNet (L1+L2) regressions
Implemented metrics:
MAE , RMSE , R²
Compared custom implementations with sklearn equivalents.
3. Regularization Analysis
Derived analytical solution for linear regression in vector form.
Analyzed impact of L1/L2 penalties on weight optimization.
Explained why L1 leads to sparse weights (feature selection).
4. Feature Normalization
Implemented MinMaxScaler and StandardScaler (both custom and sklearn)
Applied normalization before model training.
5. Non-linear Modeling
Created polynomial features (degree=10) from basic features to simulate non-linearity.
Trained all models on polynomial features to observe overfitting and regularization effects.
Tuned alpha parameters for regularized models.
6. Naive Baseline Models
Calculated mean and median baselines for performance comparison.
7. Evaluation & Comparison
Collected performance metrics across all models:
Training vs Test: MAE, RMSE, R²
Identified best and most stable models based on metric consistency.
8. Advanced Techniques (Optional)
Explored target transformation (log scaling) for heavy-tailed distributions.
Considered outlier removal in training data only.
Suggested batch training implementation as an extension.