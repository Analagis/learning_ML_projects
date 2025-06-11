## Contents. Questions from which tasks from part IV:

1. [Task 1](#task-1) \
    1.1 [Derive an analytical solution to the regression problem](#derive-an-analytical-solution-to-the-regression-problem)  
    1.2 [What changes in the solution when L1 and L2 regularizations are added to the loss function](#What-changes-in-the-solution-when-L1-and-L2-regularizations-are-added-to-the-loss-function)  
    1.3 [Explain why L1 regularization is often used to select features](#explain-why-L1-regularization-is-often-used-to-select-features)  
    1.4 [Explain how you can use the same models but make it possible to fit nonlinear dependencies](#Explain-how-you-can-use-the-same-models-but-make-it-possible-to-fit-nonlinear-dependencies)
2. [Task 7](#task-7)
2. [Task 11](#task-11)

## Task 1
### Derive an analytical solution to the regression problem
*Use a vector form of the equation.

The linear regression model in vector form is: $y = Xw+e$  
Where:
* y is the target vector $(n×1)$
* X is the feature matrix $(n×d)$
* w is the weight vector $(d×1)$ 
* e is the error term

Loss function (MSE): $L(w) = ||y-Xw||_2^2$ -> get the gradient along w -> equate it to zero ->  
-> get analytical solution (normal equation): $w* = (X^TX)^{-1}X^Ty$. Assumes $X^TX$ is invertible.

### What changes in the solution when L1 and L2 regularizations are added to the loss function.
1. L2 (Ridge) Regularization - shrinks weights but rarely sets them to zero  
    1.1 Adding regularaze parametr to loss function: $L(w) = ||y-Xw||_2^2 + lambda||w||_2^2$  
    1.2 Changing analytical solution: $w* = (X^TX+lambdaI)^{-1}X^Ty$. $lambdaI$ making $(X^TX+lambdaI)$ always invertible.
2. L1 (Lasso) Regularization - rroduces sparse weights (many exactly zero)  
    2.1 Adding regularaze parametr to loss function: $L(w) = ||y-Xw||_2^2 + lambda||w||_1$  
    2.2 Solved via proximal methods couse its not differentiable in zero
### Explain why L1 regularization is often used to select features
And why are there many weights equal to 0 after the model is fit?  

    L1 regularization imposes a geometric constraint (diamond-shaped feasible region) that tends to push weights to zero at corners of the constraint boundary. In another world, every feature increse penalty so weak feature should be zero to minimaze loss function.
### Explain how you can use the same models but make it possible to fit nonlinear dependencies
*(Linear regression, Ridge, etc.)  

    In basic case you can apply nonlinear functions to a feature. For example, polynomial function: $y = w_0+w_1x+w_2x^2+...+w_kx^k$. As you can see, equation still linear in w despite using nonliniar function on a feature.

## Task 7
### Write several examples of why and where feature normalization is mandatory and vice versa.
    Mandatary - when Algorithms which relay on scale of variables:  
        - Distance-based algorihms (KNN, K-means and etc) - features on larger scales dominate the distance calculations.\
        - PCA - prioritizes directions with maximum variance.\
        - Gradient Descent Optimization -  uneven feature scales cause slow/uneven convergence.\
        - L1/L2 - regularization penalizes large weights. Unnormalized features lead to biased penalties.\
    Optional - Algorithms doesnt relay on scales of features:\
        - Tree-Based Models - splits are based on feature thresholds, not distances or weights.\
        - Naive Bayes - uses probability distributions, not feature scales.\

A mathematical formula for MinMaxScaler: 
$$
\frac{X_i-X_{min}}{X_{max}-X_{min}}
$$
A mathematical formula for StandardScaler: 
$$
\frac{X_i-X_{mean}}{X_{std}}
$$

## Task 11
### Why outliers were removed from the training sample only?
* Preserving Real-World Test Conditions
* Avoiding Data Leakage
* Model Robustness