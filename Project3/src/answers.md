1. [Leave-One-Out Cross-Validation](#Leave-One-Out-Cross-Validation)
2. [Hyperparameter Optimization Methods](#Hyperparameter-Optimization-Methods)
3. [Feature Selection & Methods](#Feature-Selection-&-Methods)

## Leave-One-Out Cross-Validation
```
Exhaustive CV where each sample is used once as a test set (1 sample test, n-1 train)
```
* **Strengths**: Unbiased estimate (uses all data) & Low variance (nearly identical train-test sets sizes)
* **Limitations**: Computationally expensive (n fits for n samples) & High variance in performance estimate for small datasets

## Hyperparameter Optimization Methods

* **Grid Search**: Exhaustive search over all parameter combinations in a predefined grid
* **Randomized Search**: Samples parameter combinations randomly (fixed number of iterations)
* **Bayesian Optimization**:
    * Builds probabilistic model (surrogate) of objective function
    * Uses acquisition function to select next params (balance exploration/exploitation)
    * More efficient than grid/random for expensive evaluations

## Feature Selection & Methods

### Classification of Methods:

* **Filter**: Pre-select features (e.g., Pearson, Chi2) - fast but ignores model
* **Wrapper**: Use model performance (e.g., RFE) - expensive but accurate
* **Embedded**: Built into model training (e.g., Lasso) - balanced approach

### Some Methods:

* **Pearson Correlation**:
    * Measures linear dependence (-1 to 1)
    * Assumes normal distribution, sensitive to outliers
* **Chi-squared**:
    * Tests independence between categorical features and target
    * Requires non-negative values and sufficient sample size
* **Lasso (L1 Regularization)**:
    * Adds penalty term ($λ∑|wᵢ|$) to linear model
    * Shrinks weak features to exactly zero
    * Automatic feature selection during training
* **Permutation Importance**:
    * Randomly shuffle feature values
    * Measure performance drop
    * Significant drop → important feature
* **SHAP (SHapley Additive exPlanations)**:
    * Game theory approach for feature importance
    * Shows magnitude/direction of feature impact
    * Model-agnostic and consistent