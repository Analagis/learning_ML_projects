import re
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCreating:

    @staticmethod
    def column_clearing(column):
        """
        Removes unnecessary characters from string values in a DataFrame column.

        Parameters
        ----------
        column : pandas.Series
            Column containing string values.

        Returns
        -------
        pandas.Series
            Cleaned column.
        """
        return column.apply(lambda x: re.sub(r'[ \[\]\'\"]', '', str(x)))
    
    @staticmethod
    def columns_creating(features, df):
        """
        Creates new binary features in the DataFrame based on presence of elements in the 'features' column.

        Parameters
        ----------
        features : list of str
            List of feature names to create.
        df : pandas.DataFrame
            Original DataFrame with a 'features' column.
        """
        for feature in features:
            df[feature] = df['features'].apply(lambda x: 1 if feature in x else 0)

    @staticmethod
    def remove_outliers_iqr(df, column):
        """
        Removes outliers from a specified column using the IQR method.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
        column : str
            Name of the column to remove outliers from.

        Returns
        -------
        pandas.DataFrame
            DataFrame without outliers.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

class ModelsEval:
    def __init__(self):
        """
        Initializes result containers for model evaluation metrics.
        """
        self.result_MAE = pd.DataFrame(columns=['model', 'train', 'test'])
        self.result_RMSE = pd.DataFrame(columns=['model', 'train', 'test'])
        self.result_R2 = pd.DataFrame(columns=['model', 'train', 'test'])
        pd.set_option('display.float_format', '{:.1f}'.format)

    def insert_eval_in_DF(self, model, y_true_train, y_pred_train, y_true_test, y_pred_test):
        """
        Inserts evaluation metrics into internal DataFrames.

        Parameters
        ----------
        model : str
            Model name.
        y_true_train : array-like
            True target values for training data.
        y_pred_train : array-like
            Predicted values for training data.
        y_true_test : array-like
            True target values for test data.
        y_pred_test : array-like
            Predicted values for test data.
        """
        self.result_MAE.loc[len(self.result_MAE)] = {"model": model, "train": mean_absolute_error(y_true_train, y_pred_train),
                            "test": mean_absolute_error(y_true_test, y_pred_test)}
        self.result_RMSE.loc[len(self.result_RMSE)] = {"model": model, "train": root_mean_squared_error(y_true_train, y_pred_train),
                            "test": root_mean_squared_error(y_true_test, y_pred_test)}
        self.result_R2.loc[len(self.result_R2)] = {"model": model, "train": r2_score(y_true_train, y_pred_train) * 100,
                            "test": r2_score(y_true_test, y_pred_test) * 100}


    def find_best_alpha(self, X_train, X_test, y_train, y_test, alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0]):
        """
        Finds the best regularization parameter (alpha) for Ridge, Lasso and ElasticNet models.

        Parameters
        ----------
        X_train : array-like
            Training feature matrix.
        X_test : array-like
            Test feature matrix.
        y_train : array-like
            Target values for training.
        y_test : array-like
            Target values for testing.
        alpha_range : list of float, optional
            List of alpha values to test.

        Returns
        -------
        dict
            Best model info including name, parameters, and metrics.
        """

        models = {
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet()
        }

        best_metrics = {
            'RMSE': np.inf,
            'R2': -np.inf,
            'model_params': None
        }
        
        best_model_info = None
        
        for alpha in alpha_range:
            for model_name, model in models.items():
                # Создаем модель с текущим alpha
                model_instance = model.__class__(alpha=alpha)
                model_instance.fit(X_train, y_train)
                y_pred_test = model_instance.predict(X_test)
                
                # Метрики
                rmse_test = root_mean_squared_error(y_test, y_pred_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                r2_test = r2_score(y_test, y_pred_test)
                
                # Обновляем лучший результат по RMSE и R2
                if rmse_test < best_metrics['RMSE']:
                    best_metrics['RMSE'] = rmse_test
                    best_metrics['R2'] = r2_test
                    best_metrics['MAE'] = mae_test
                    best_model_info = {
                        'model_name': model_name,
                        'alpha': alpha,
                        'model': model_instance,
                        'rmse': rmse_test,
                        'mae': mae_test,
                        'r2': r2_test
                    }

        if best_model_info:
            reg = best_model_info["model"]
            reg.set_params(alpha = best_model_info['alpha'])
            self.insert_eval_in_DF(best_model_info['model_name']+f"_alpha_{best_model_info['alpha']}", y_train, reg.predict(X_train), y_test, reg.predict(X_test))

        return best_model_info

    def run_models(self, X_train,X_test, y_train, y_test, type):
        """
        Trains and evaluates multiple regression models on the provided train and test datasets.
        
        Supported models: Linear Regression, Ridge, Lasso, ElasticNet.
        Evaluation results are inserted into internal result DataFrames via insert_eval_in_DF method.
        
        Args:
            X_train (array-like): Training input samples.
            X_test (array-like): Test input samples.
            y_train (array-like): Target values for training.
            y_test (array-like): Target values for testing.
            type (str): Type or identifier of data being processed (e.g., 'scaled', 'raw').
        """
        reg = LinearRegression().fit(X_train, y_train)
        self.insert_eval_in_DF("linear_regression_"+type, y_train, reg.predict(X_train), y_test, reg.predict(X_test))
        reg = Ridge().fit(X_train, y_train)
        self.insert_eval_in_DF("ridge_"+type, y_train, reg.predict(X_train), y_test, reg.predict(X_test))
        reg = Lasso().fit(X_train, y_train)
        self.insert_eval_in_DF("lasso_"+type, y_train, reg.predict(X_train), y_test, reg.predict(X_test))
        reg = ElasticNet().fit(X_train, y_train)
        self.insert_eval_in_DF("elastic_net_"+type, y_train, reg.predict(X_train), y_test, reg.predict(X_test))

    def show_results(self):
        """
        Prints all stored evaluation results for MAE, RMSE, and R2 metrics.
        Results are printed as DataFrames for easy interpretation.
        """
        print("MAE Results:")
        print(self.result_MAE)
        print("\nRMSE Results:")
        print(self.result_RMSE)
        print("\nR2 Results (in %):")
        print(self.result_R2)

    def sort_df(self, df_old, asc = True):
        """
        Sorts a DataFrame by test score, difference between test and train scores,
        and then by train score.

        Args:
            df_old (pd.DataFrame): Input DataFrame to be sorted.
            asc (bool): Whether to sort in ascending order (True) or descending (False).

        Returns:
            pd.DataFrame: Sorted DataFrame with an added 'diff' column.
        """
        df = df_old.copy()
        df["diff"] = abs(df['test'] - df['train'])
        df = df.sort_values(by=['test', 'diff', 'train'], ascending=[asc, True, asc])
        return df

    def show_best_models(self, n_top = 5):
        """
        Prints top N best performing models based on evaluation metrics.
        For R2, sorting is done in descending order.

        Args:
            n_top (int): Number of top models to display.
        """
        print("MAE Results:")
        print(self.sort_df(self.result_MAE).head(n_top))
        print("\nRMSE Results:")
        print(self.sort_df(self.result_RMSE).head(n_top))
        print("\nR2 Results (in %):")
        print(self.sort_df(self.result_R2, asc = False).head(n_top))
        

# Абстрактный класс для оптимизаторов
class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms used in regression models.
    
    Subclasses must implement the `optimize` method which returns model weights and bias.
    """
    
    @abstractmethod
    def optimize(self, X, y, **kwargs):
        """Вернуть параметры: weights и bias"""
        pass

# Batch
class BGDOptimizer(BaseOptimizer):
    """
    Batch Gradient Descent optimizer with optional regularization.

    Args:
        learning_rate (float): Step size for weight updates.
        n_iter (int): Maximum number of iterations.
        regularization_type (str): Regularization method ('none', 'l1', 'l2', 'elasticnet').
        alpha (float): Regularization strength.
        l1_ratio (float): Ratio of L1 penalty in ElasticNet (0 <= l1_ratio <= 1).
    """
    def optimize(self, X, y, learning_rate=0.01, n_iter=10,
                    regularization_type='none', alpha=0.0, l1_ratio=0.5):
        X = np.array(X)
        y = np.array(y).ravel()
        n_samples, n_features = X.shape
        
        weights = np.random.randn(n_features) * 0.01
        bias = 0

        tol = 1e-4
        prev_loss = np.inf
        
        for i in range(n_iter):
            predictions = X.dot(weights) + bias
            errors = predictions - y

            current_loss = np.mean((predictions - y)**2)
            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss
            
            dw = (2 / n_samples) * X.T.dot(errors)
            db = (2 / n_samples) * np.sum(errors)

            if regularization_type == 'l2':
                dw += 2*alpha*weights
            elif regularization_type == 'l1':
                dw += 2*alpha*np.sign(weights)
            elif regularization_type=='elasticnet':
                dw += 2*alpha*(l1_ratio*np.sign(weights)+(1-l1_ratio)*weights)

            weights -= learning_rate*dw
            bias -= learning_rate*db
        
        return weights, bias

class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer with optional regularization.

    Args:
        learning_rate (float): Initial learning rate.
        n_iter (int): Number of epochs.
        regularization_type (str): Regularization method ('none', 'l1', 'l2', 'elasticnet').
        alpha (float): Regularization strength.
        l1_ratio (float): Ratio of L1 penalty in ElasticNet.
    """
    def optimize(self, X, y, learning_rate=0.1, n_iter=10,
                 regularization_type='none', alpha=0.0, l1_ratio=0.5):
        X = np.array(X)
        y = np.array(y).ravel()
        n_samples, n_features = X.shape
        
        if n_features > 1:
            weights = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            weights = np.array([np.cov(X.flatten(), y)[0,1] / np.var(X)])

        bias = 0

        initial_lr = learning_rate
        decay_rate = 0.01

        for iteration in range(n_iter):
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                prediction = np.dot(x_i, weights) + bias
                error = prediction - y_i

                learning_rate = initial_lr / (1 + iteration * decay_rate)
                
                dw = 2 * error * x_i
                db = 2 * error
                
                # Регуляризация
                if regularization_type == 'l2':
                    dw += 2 * alpha * weights
                elif regularization_type == 'l1':
                    dw += 2 * alpha * np.sign(weights)
                elif regularization_type == 'elasticnet':
                    dw += 2 * alpha * (l1_ratio*np.sign(weights) + (1 - l1_ratio)*weights)

                # Обновление веса и смещения
                weights -= learning_rate*dw
                bias -= learning_rate*db
                
        return weights, bias

# Аналитическое решение с регуляризацией
class NormalEquationOptimizer(BaseOptimizer):
    """
    Analytical solution to linear regression using normal equation.
    Supports L2 regularization only.

    Raises:
        NotImplementedError: If attempting to use L1 or ElasticNet regularization.
    """
    def optimize(self, X, y, **kwargs):
        alpha = kwargs.get('alpha', 0.0)
        regularization_type = kwargs.get('regularization_type', 'none')

        X = np.array(X)
        y = np.array(y).ravel()

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        if regularization_type == 'l2':
            reg_matrix = alpha * np.eye(X_b.shape[1])
            reg_matrix[0, 0] = 0  # не регулируем свободный член
            theta = np.linalg.inv(X_b.T @ X_b + reg_matrix) @ X_b.T @ y
            
        elif regularization_type == 'elasticnet':
            # ElasticNet не имеет точного аналитического решения
            raise NotImplementedError("ElasticNet regularization requires iterative solvers.")
            
        elif regularization_type == 'l1':
            # Аналитического решения для L1 нет
            raise NotImplementedError("L1 regularization requires iterative solvers.")
        
        else:
            # Без регуляризации
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        
        bias = theta[0]
        weights = theta[1:]
        return weights, bias

# Базовый класс модели регрессии
class BaseRegressionModel(BaseEstimator):
    """
    Base class for regression models using customizable optimizer.

    Args:
        regularization_type (str): Type of regularization ('none', 'l1', 'l2', 'elasticnet').
        optimizer (BaseOptimizer): Optimizer instance to use for training.
        learning_rate (float): Learning rate for optimizer.
        n_iter (int): Number of iterations for optimizer.
        alpha (float): Regularization parameter.
        l1_ratio (float): Mixing parameter for ElasticNet regularization.
    """
    def __init__(self,regularization_type='none',  optimizer = BGDOptimizer(), learning_rate=0.01, n_iter=10,
                 alpha=1.0, l1_ratio=0.5):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.regularization_type = regularization_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        self.weights, self.bias = self.optimizer.optimize(
            X, y,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            regularization_type=self.regularization_type,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio
        )
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class S21LinearRegression(BaseRegressionModel):
    """
    Linear regression model without regularization.
    Uses Batch Gradient Descent by default.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
class S21RidgeRegression(BaseRegressionModel):
    """
    Ridge regression model with L2 regularization.
    Uses Batch Gradient Descent by default.
    """
    def __init__(self, **kwargs):
        super().__init__(regularization_type="l2", **kwargs)
        
class S21LassoRegression(BaseRegressionModel):
    """
    Lasso regression model with L1 regularization.
    Uses Batch Gradient Descent by default.
    """
    def __init__(self, **kwargs):
        super().__init__(regularization_type="l1", **kwargs)

class S21ElasticNetRegression(BaseRegressionModel):
    """
    ElasticNet regression model with both L1 and L2 regularization.
    Uses Batch Gradient Descent by default.
    """
    def __init__(self, **kwargs):
        super().__init__(regularization_type = "elasticnet", **kwargs)

class S21MinMaxScaler(TransformerMixin):
    """
    Transforms features by scaling each feature to a given range (default [0, 1]).

    Args:
        feature_range (tuple): Desired range of transformed data (min, max).
    """
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.feature_range = feature_range
    
    def fit(self, X):
        X = np.array(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1e-8 # Предотвращение деления на ноль
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_

        return self
    
    def transform(self, X):
        X = np.array(X)
        return X * self.scale_ + self.min_

class S21StandardScaler(TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        X = np.array(X)
        self.mean_ = X.mean(axis=0)
        std_dev = X.std(axis=0)
        std_dev[std_dev == 0] = 1e-8
        self.scale_ = std_dev

        return self
    
    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_