## Project Summary
This project implements advanced validation schemes, hyperparameter tuning methods, and feature selection techniques for a real estate dataset (from Kaggle). The focus is on model validation strategies, Lasso-based and correlation-based feature selection, and optimization using Grid Search, Random Search, and Optuna.

📌 Contents
1. Theory & Questions
Leave-One-Out Cross-Validation (LOO) :
Strengths: low bias, uses all data.
Weaknesses: high variance, computationally expensive.
Hyperparameter Tuning :
Grid Search : exhaustive search over parameter grid.
Randomized Search : samples random combinations efficiently.
Bayesian Optimization : uses probabilistic models to find optimal params.
Feature Selection Methods :
Filter: Pearson (linear correlation), Chi2 (categorical relevance).
Wrapper: Lasso (L1 regularization for sparsity).
Permutation importance: evaluates feature impact by shuffling.
Model-based: SHAP values for interpretability.
2. Data Preprocessing
Loaded train/test datasets.
Preprocessed Interest Level categorical feature.
Created 20 binary features from the Features column:
'Elevator', 'HardwoodFloors', 'CatsAllowed', ..., 'Terrace'.
Final feature set includes:
bathrooms, bedrooms, interest_level, and 20 new binary features.

3. Data Splitting Methods
Custom implementations of:

Simple train-test split (test_size)
Train-validation-test split (val_size, test_size)
Date-based splits using date_split, validation_date, test_date
4. Cross-Validation Schemes
Implemented and compared with sklearn:

K-Fold CV
Grouped K-Fold CV (group_field)
Stratified K-Fold CV (stratify_field)
TimeSeriesSplit (date_field)
Each returns list of (train_indices, test_indices).

5. Validation Comparison
Applied all implemented and sklearn validation methods.
Compared resulting training data distributions.
Evaluated which scheme best reflects real-world performance.
Chose the most appropriate validation method based on task type (e.g., time-aware vs. random).
6. Feature Selection
Used Lasso regression on normalized data with 60/20/20 train/val/test split.
Ranked features by weights, refit on top 10.
Implemented and applied:
Correlation + NaN ratio filtering , refit on top 10.
Permutation importance , refit on top 10.
SHAP values , used to select top features.
Compared results across:
Model performance (MAE, RMSE, R²)
Stability
Speed
7. Hyperparameter Optimization
Implemented:
Grid Search
Random Search
For alpha and l1_ratio in ElasticNet
Fitted best model and evaluated metrics.
Used Optuna for Bayesian optimization.
Ran Optuna within a cross-validation loop.
Compared all approaches in terms of speed and performance.