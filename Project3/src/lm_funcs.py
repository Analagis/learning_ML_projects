import re
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import ks_2samp
from itertools import product
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pointbiserialr, spearmanr
from joblib import Parallel, delayed
from sklearn.base import clone





class FeatureCreating:

    @staticmethod
    def column_clearing(column):
        return column.apply(lambda x: re.sub(r'[ \[\]\'\"]', '', str(x)))
    
    @staticmethod
    def columns_creating(features, df):
        for feature in features:
            df[feature] = df['features'].apply(lambda x: 1 if feature in x else 0)


class ModelsEval:
    def __init__(self):
        self.result_MAE = pd.DataFrame(columns=['model', 'train', "val", 'test'])
        self.result_RMSE = pd.DataFrame(columns=['model', 'train', "val", 'test'])
        self.result_R2 = pd.DataFrame(columns=['model', 'train', "val",'test'])
        self.find_fs_time = pd.DataFrame(columns=['model', 'find_fs_time'])
        pd.set_option('display.float_format', '{:.2f}'.format)

    def insert_eval_in_DF(self, model, y_true_train, y_pred_train, y_true_val, y_pred_val, y_true_test, y_pred_test, time_count):
        self.result_MAE.loc[len(self.result_MAE)] = {"model": model, "train": mean_absolute_error(y_true_train, y_pred_train), 
                                                     "val": mean_absolute_error(y_true_val, y_pred_val),
                            "test": mean_absolute_error(y_true_test, y_pred_test)}
        self.result_RMSE.loc[len(self.result_RMSE)] = {"model": model, "train": root_mean_squared_error(y_true_train, y_pred_train), 
                                                       "val": root_mean_squared_error(y_true_val, y_pred_val),
                            "test": root_mean_squared_error(y_true_test, y_pred_test)}
        self.result_R2.loc[len(self.result_R2)] = {"model": model, "train": r2_score(y_true_train, y_pred_train) * 100,
                                                   "val": r2_score(y_true_val, y_pred_val) * 100,
                            "test": r2_score(y_true_test, y_pred_test) * 100}
        self.find_fs_time.loc[len(self.find_fs_time)] = {"model": model, "find_fs_time": round(time_count,2)}

    def fit_model_with_evaluation(self, model_name, X_train, y_train, X_validation, y_validation, X_test, y_test, 
                                  training_time, slice, model_class=Lasso, model_params=None):
        if model_params is None:
            model_params = {}
        model = model_class(**model_params)
        model.fit(X_train[slice], y_train)
        
        self.insert_eval_in_DF(model_name, y_train, model.predict(X_train[slice]),
                                        y_validation, model.predict(X_validation[slice]),
                                        y_test, model.predict(X_test[slice]), training_time)
        
    def show_results(self):
        print("MAE Results:")
        print(self.result_MAE)
        print("\nRMSE Results:")
        print(self.result_RMSE)
        print("\nR2 Results (in %):")
        print(self.result_R2)
        print("\nFind features timing (in sec):")
        print(self.find_fs_time)

    def rank_models(self, weights=None):
        """
        Ранжирует модели по качеству прогноза, стабильности и времени.

        Parameters:
        mae_df, rmse_df, r2_df: DataFrame с метриками ('model', 'train', 'val', 'test')
        time_df: DataFrame с временем ('model', 'find_fs_time')
        weights: dict с весами {'quality': ..., 'stability': ..., 'time': ...}

        Returns:
        pd.DataFrame — модели в порядке убывания качества
        """

        if weights is None:
            weights = {'quality': 0.5, 'stability': 0.3, 'time': 0.2}

        # Объединяем всё в один DataFrame
        df = pd.merge(self.result_MAE[['model', 'train', 'val', 'test']], self.find_fs_time, on='model')
        df = df.rename(columns={'train': 'mae_train', 'val': 'mae_val', 'test': 'mae_test'})

        df['rmse_train'] = self.result_RMSE['train'].values
        df['rmse_val'] = self.result_RMSE['val'].values
        df['rmse_test'] = self.result_RMSE['test'].values

        df['r2_train'] = self.result_R2['train'].values
        df['r2_val'] = self.result_R2['val'].values
        df['r2_test'] = self.result_R2['test'].values

        # Нормализация метрик (чем меньше, тем лучше для MAE/RMSE)
        for col in ['mae_train', 'rmse_train']:
            df[col + '_norm'] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Для R2: чем больше, тем лучше
        for col in ['r2_train']:
            df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Качество прогноза: среднее значение нормализованных метрик
        quality_cols = [c + '_norm' for c in ['mae_train', 'rmse_train', 'r2_train']]
        df['train_quality_score_norm'] = df[quality_cols].median(axis=1)

        # Стабильность: средняя разница между train-val и val-test
        df['stability_score'] = (
            abs(df['rmse_train'] - df['rmse_val']) +
            abs(df['rmse_val'] - df['rmse_test'])
        ) / 2

        # Нормализуем стабильность (меньше → лучше, переворачиваем в больше -> лучше)
        df['stability_score_rmse_norm'] = 1 - (df['stability_score'] - df['stability_score'].min()) / \
                                    (df['stability_score'].max() - df['stability_score'].min())

        # Время: нормализация (меньше → лучше, переворачиваем в больше -> лучше)
        df['time_score_norm'] = 1 - (df['find_fs_time'] - df['find_fs_time'].min()) / \
                                (df['find_fs_time'].max() - df['find_fs_time'].min())

        # Общий балл
        df['total_score'] = (
            weights['quality'] * df['train_quality_score_norm'] +
            weights['stability'] * df['stability_score_rmse_norm'] +
            weights['time'] * df['time_score_norm']
        )

        # Сортируем по убыванию
        ranked_df = df.sort_values(by='total_score', ascending=False).reset_index(drop=True)

        return ranked_df[['model', 'total_score', 'train_quality_score_norm', 'stability_score_rmse_norm', 'time_score_norm']]


class Splits:

    @staticmethod
    def random_split_two(data, test_size=0.2, random_state=21):

        if random_state is not None:
            np.random.seed(random_state)
            
        if isinstance(data, pd.DataFrame):
            indices = np.arange(len(data))
            test_indices = np.random.choice(indices, size=int(len(data)*test_size), replace=False)
            train_indices = np.setdiff1d(indices, test_indices)
            
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
        else:
            indices = np.arange(data.shape[0])
            test_indices = np.random.choice(indices, size=int(data.shape[0]*test_size), replace=False)
            train_indices = np.setdiff1d(indices, test_indices)
            
            train_data = data[train_indices]
            test_data = data[test_indices]
            
        return train_data, test_data
    
    @staticmethod
    def random_split_three(data, test_size=0.2, validation_size=0.2, random_state=21):
        if random_state is not None:
            np.random.seed(random_state)
            
        total_test_size = test_size + validation_size
        train, temp = Splits.random_split_two(data, test_size=total_test_size, random_state=random_state)
        
        relative_val_size = validation_size / total_test_size
        validation, test = Splits.random_split_two(temp, test_size=1-relative_val_size, random_state=random_state)
        
        return train, validation, test
    
    @staticmethod
    def date_split_two(data, date_col, date_split):
        train = data[data[date_col] < date_split]
        test = data[data[date_col] >= date_split]
        return train, test
    
    @staticmethod
    def date_split_three(data, date_col, validation_date, test_date):
        train = data[data[date_col] < validation_date]
        validation = data[(data[date_col] >= validation_date) & (data[date_col] < test_date)]
        test = data[data[date_col] >= test_date]
        if not (train[date_col].max() < validation[date_col].max() < test[date_col].max()):
            raise ValueError("The time order has been violated: train < validation < test")
        return train, validation, test
    
class CrossValidation:

    @staticmethod
    def k_fold(data, k = 5, shuffle = False, random_state = None):
        """
        Стандартный K-Fold split.
        
        Параметры:
        - data: входные данные
        - k: количество фолдов
        - shuffle: перемешивать ли данные перед разбиением
        - random_state: seed для воспроизводимости
        
        Возвращает:
        - Список из k кортежей (train_indices, test_indices)
        """
        if isinstance(data, pd.DataFrame):
            n_samples = len(data)
        else:
            n_samples = data.shape[0]
            
        indices = np.arange(n_samples)
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
            
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        
        current = 0
        splits = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            splits.append((train_idx, test_idx))
            current = stop
            
        return splits
    
    @staticmethod
    def grouped_k_fold(data, k = 5, group_field = None, shuffle = False, random_state = None):
        """
        Group K-Fold split. Обеспечивает, что одна группа не попадает 
        одновременно в train и test.
        Используется жадное распределение для равномерного распределения наблюдений 
        
        Параметры:
        - data: DataFrame с данными
        - k: количество фолдов
        - group_field: поле для группировки
        - shuffle: перемешивать ли группы перед разбиением
        - random_state: seed для воспроизводимости
        
        Возвращает:
        - Список из k кортежей (train_indices, test_indices)
        """
        if group_field is None:
            raise ValueError("group_field должен быть указан")
            
        groups = data[group_field].unique()

        if len(groups) < k:
            raise ValueError("num og group should be more or equile to k")

        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(groups)
        else:
            group_sizes = data.groupby(group_field).size()
            groups = group_sizes.sort_values(ascending=False).index.values

        group_to_fold = {}
        fold_sizes = [0] * k
        for group in groups:
            count = (data[group_field] == group).sum()
            fold_idx = np.argmin([fold_sizes[i] + count for i in range(k)])
            group_to_fold[group] = fold_idx
            fold_sizes[fold_idx] += count

        splits = []
        for i in range(k):
            test_mask = data[group_field].apply(lambda x: group_to_fold.get(x, -1) == i)
            train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]
            splits.append((train_idx, test_idx))

        return splits

    @staticmethod
    def stratified_k_fold(data, k = 5, stratify_field = None, shuffle = False, random_state = None):
        """
        Stratified K-Fold split. Сохраняет распределение классов в каждом фолде.
        
        Параметры:
        - data: DataFrame с данными
        - k: количество фолдов
        - stratify_field: поле для стратификации
        - shuffle: перемешивать ли данные внутри классов перед разбиением
        - random_state: seed для воспроизводимости
        
        Возвращает:
        - Список из k кортежей (train_indices, test_indices)
        """
        if stratify_field is None:
            raise ValueError("stratify_field должен быть указан")
        
        if shuffle and random_state is not None:
            np.random.seed(random_state)
            
        class_indices = defaultdict(list)
        for idx, class_val in enumerate(data[stratify_field]):
            class_indices[class_val].append(idx)
            
        splits = []
        
        class_fold_indices = defaultdict(list)

        for class_val, indices in class_indices.items():
            if shuffle:
                indices = list(indices)
                np.random.shuffle(indices)

            fold_sizes = np.full(k, len(indices) // k, dtype=int)
            fold_sizes[:len(indices) % k] += 1

            current = 0
            for i in range(k):
                start, stop = current, current + fold_sizes[i]
                class_fold_indices[class_val].append(indices[start:stop])
                current = stop

        for i in range(k):
            test_idx = []
            for class_val in class_indices:
                test_idx.extend(class_fold_indices[class_val][i])
            test_idx = np.array(test_idx)

            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)

            splits.append((train_idx, test_idx))

        return splits
    
    @staticmethod
    def time_series_split(data, k = 5, date_field = None):
        """
        Time Series split. Обеспечивает временной порядок - train всегда до test.
        
        Параметры:
        - data: DataFrame с данными
        - k: количество фолдов
        - date_field: поле с датой/временем
        
        Возвращает:
        - Список из k кортежей (train_indices, test_indices)
        """
        if date_field is None:
            raise ValueError("date_field должен быть указан")
            
        data_sorted = data.sort_values(date_field).copy()
        n_samples = len(data_sorted)
        min_train_size = n_samples // (k + 1)
        
        splits = []
        for i in range(1, k + 1):
            train_size = min_train_size * i
            test_size = min(n_samples - train_size, min_train_size)
            
            if test_size == 0:
                continue
                
            train_idx = np.arange(train_size)
            test_idx = np.arange(train_size, train_size + test_size)
            splits.append((train_idx, test_idx))
            
        return splits
    
class Comparasion:

    @staticmethod
    def compare_indices(custom_train_idx, sklearn_train_idx):
        intersection = np.intersect1d(custom_train_idx, sklearn_train_idx)
        union = np.union1d(custom_train_idx, sklearn_train_idx)
        jaccard = len(intersection) / len(union)
        overlap = len(intersection) / len(custom_train_idx)
        return {
            'jaccard': jaccard,
            'overlap': overlap
        }
    
    @staticmethod
    def loop_for_compare_index(custom_list_indices, sk_list_indices):
        results = []
        for i, ((custom_train_idx, custom_test_idx), (sklearn_train_idx, sklearn_test_idx)) in enumerate(zip(custom_list_indices, sk_list_indices)):
            metrics = Comparasion.compare_indices(custom_train_idx, sklearn_train_idx)
            metrics['fold'] = i+1
            results.append(metrics)

        return pd.DataFrame(results)
    
    @staticmethod
    def compare_all_features_distributions(X_custom, X_sklearn):
        results = []
        for col in X_custom.columns:
            stat, p_val = ks_2samp(X_custom[col], X_sklearn[col])
            results.append({
                'feature': col,
                'ks_stat': stat,
                'p_value': p_val,
                'same_dist': p_val > 0.05
            })

        df_results = pd.DataFrame(results)

        same_ratio = df_results['same_dist'].mean()

        all_same = df_results['same_dist'].all()

        return {
            'same_distribution_share': same_ratio,
            'all_same': all_same
        }

    @staticmethod
    def loop_for_compare_distributions(custom_list_indices, sk_list_indices, df):
        results = []
        for i, ((custom_train_idx, custom_test_idx), (sklearn_train_idx, sklearn_test_idx)) in enumerate(zip(custom_list_indices, sk_list_indices)):
            metrics = Comparasion.compare_all_features_distributions(df.iloc[custom_train_idx], df.iloc[sklearn_train_idx])
            metrics['fold'] = i+1
            results.append(metrics)

        return pd.DataFrame(results)
    

class FeatureSelection:
    @staticmethod
    def select_features_by_nan_corr(X, y, max_nan_ratio=0.1, min_corr=0.1, n_features=10):
        """
        Отбирает признаки по двум критериям:
        1. Максимальная доля пропусков (max_nan_ratio)
        2. Минимальная корреляция с целевой переменной (min_corr)
        При необходимости ослабляет условия, чтобы вернуть n_features штук.

        Parameters:
        X : pd.DataFrame, np.ndarray, list of lists или pd.Series
            Признаки
        y : pd.Series, np.ndarray
            Целевая переменная
        max_nan_ratio : float
            Максимальная доля NaN в признаке
        min_corr : float
            Минимальное значение корреляции (по модулю)
        n_features : int
            Желаемое число признаков

        Returns:
        list
            Список названий или индексов отобранных признаков
        """

        # Приведение к DataFrame
        if isinstance(X, list):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif isinstance(X, pd.Series):
            X = X.to_frame()

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Проверка совместимости
        if len(X) != len(y):
            raise ValueError("Длина X и y должна совпадать")

        # Шаг 1: Фильтрация по nan_ratio
        nan_ratios = X.isna().mean()
        valid_features = nan_ratios[nan_ratios <= max_nan_ratio].index

        if len(valid_features) == 0:
            raise ValueError("Нет признаков, удовлетворяющих заданному nan_ratio")

        X_valid = X[valid_features]

        # Шаг 2: Фильтрация по корреляции
        correlations = X_valid.apply(
            lambda col: abs(pointbiserialr(col, y)[0])
            if col.nunique(dropna=True) <= 2
            else abs(spearmanr(col, y)[0])
            )
        correlated_features = correlations[correlations >= min_corr].sort_values(ascending=False).index

        if len(correlated_features) == 0:
            raise ValueError("Нет признаков, удовлетворяющих заданному уровню корреляции")

        # Шаг 3: Жадный отбор с учетом мультиколлинеарности
        selected = []
        remaining = correlated_features.copy()

        while len(selected) < n_features and len(remaining) > 0:
            # Первый — с наибольшей корреляцией
            if not selected:
                best = remaining[0]
                selected.append(best)
                remaining = remaining.drop(best)
                continue

            # Оценить, какой признак добавить, чтобы он был слабо коррелирован с уже выбранными
            scores = {}
            for feat in remaining:
                corr_with_selected = X_valid[selected].corrwith(X_valid[feat]).abs().max()
                score = correlations.loc[feat] / (1 + corr_with_selected)  # снижаем ценность сильно скоррелированных
                scores[feat] = score

            best_feat = max(scores, key=scores.get)
            selected.append(best_feat)
            remaining = remaining.drop(best_feat)

        # Если не хватает признаков — ослабляем условия
        if len(selected) < n_features:
            print(f"Не удалось набрать {n_features} признаков при текущих условиях: \n-max_nan_ratio = {max_nan_ratio:.2f}; \n-min_corr = {min_corr:.2f}. "
                f"\nОслабляем фильтры и повторяем...")

            return FeatureSelection.select_features_by_nan_corr(X, y, max_nan_ratio=min(1.0, max_nan_ratio + 0.05),
                                min_corr=max(0.0, min_corr - 0.05), n_features=n_features)

        return selected
    
    @staticmethod
    def permutation_importance(model, X, y, metric=mean_absolute_error, n_repeats=5, n_features=5, random_state=None):
        """
        Вычисляет важность признаков через перестановку (Permutation Importance).

        Parameters:
        model : object
            Обученная модель с методом predict
        X : pd.DataFrame or np.ndarray
            Признаки
        y : array-like
            Целевая переменная
        metric : callable
            Функция оценки качества (чем меньше — тем лучше)
        n_repeats : int
            Сколько раз повторять перемешивание для устойчивости
        n_features : int, optional
            Сколько самых важных признаков вернуть
        random_state : int, optional
            Зерно для генератора случайных чисел

        Returns:
        pd.DataFrame
            DataFrame с признаками и их важностью (вклад в ухудшение метрики)
        """

        rng = np.random.RandomState(random_state)

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_df = X.copy()
        else:
            feature_names = [f"X_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)

        baseline_score = metric(y, model.predict(X_df))
        importance_scores = []

        for col in feature_names:
            scores = []
            for _ in range(n_repeats):
                # Копируем DataFrame и перемешиваем значения одного столбца
                X_permuted = X_df.copy()
                X_permuted[col] = rng.permutation(X_permuted[col])
                # Предсказываем и считаем ухудшение метрики
                y_pred = model.predict(X_permuted)
                permuted_score = metric(y, y_pred)
                scores.append(permuted_score - baseline_score)  # положительное значение = вред от перемешивания

            mean_score = np.mean(scores)
            importance_scores.append((col, mean_score))

        # Создаём DataFrame и сортируем по важности
        importance_df = pd.DataFrame(importance_scores, columns=["Feature", "Importance"])
        importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Возвращаем нужное количество признаков
        if n_features is not None:
            if len(importance_df) < n_features:
                raise ValueError("n_features больше, чем количество доступных признаков")
            return importance_df.head(n_features)
        return importance_df

class BaseSearchCV:
    def __init__(self, estimator, param_grid, scoring=root_mean_squared_error,
                 cv=5, minimize=True, n_jobs=None, verbose=False, refit=True):
        """
        Базовый класс для поиска гиперпараметров.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.minimize = minimize  # True если метрику нужно минимизировать
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit= refit

        self.best_params_ = None
        self.best_score_ = np.inf if minimize else -np.inf
        self.cv_results_ = []

    def fit(self, X, y):
        kf = KFold(n_splits=self.cv)

        def evaluate(params):
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = clone(self.estimator).set_params(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = self.scoring(y_val, y_pred)
                scores.append(score)

            mean_score = np.mean(scores)
            return {"params": params, "mean_score": mean_score}

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate)(params) for params in self._generate_param_combinations()
        )

        for res in results:
            self.cv_results_.append(res)
            self._update_best(res["mean_score"], res["params"])
            if self.verbose:
                print(f"Params: {res['params']} | Score: {res['mean_score']:.4f}")

        # Рефит — обучаем лучшую модель на всем датасете
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        return self

    def _generate_param_combinations(self):
        """
        Генерирует наборы параметров для перебора.
        Должен быть реализован в дочерних классах.
        """
        raise NotImplementedError

    def _update_best(self, score, params):
        """
        Обновляет лучшие параметры и оценку.
        """
        if (self.minimize and score < self.best_score_) or \
           (not self.minimize and score > self.best_score_):
            self.best_score_ = score
            self.best_params_ = params

    def get_best_params(self):
        return self.best_params_

    def get_best_score(self):
        return self.best_score_

    def get_cv_results(self):
        return self.cv_results_


class GridSearchCV(BaseSearchCV):
    def _generate_param_combinations(self):
        """
        Генерирует все возможные комбинации параметров (grid search).
        """
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        return [dict(zip(keys, v)) for v in product(*values)]


class RandomSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, random_state=None, **kwargs):
        super().__init__(estimator, param_grid=param_distributions, **kwargs)
        self.n_iter = n_iter
        self.random_state = random_state
        random.seed(random_state)

    def _generate_param_combinations(self):
        keys = list(self.param_grid.keys())
        distributions = []

        # сохранение способов получения случайных значений в зависимости от типа данных
        for key in keys:
            dist = self.param_grid[key]

            if isinstance(dist, (list, tuple)):
                # Список или кортеж — случайный выбор
                distributions.append(lambda d=dist: random.choice(d))
            elif isinstance(dist, range):
                # range -> случайный элемент
                distributions.append(lambda d=dist: random.choice(list(d)))
            elif hasattr(dist, 'rvs'):
                # Поддержка scipy.stats распределений (в т.ч. frozen)
                distributions.append(lambda d=dist: d.rvs())
            else:
                raise ValueError(f"Неизвестный тип распределения для {key}: {type(dist)}")

        return [
            dict(zip(keys, [round(d(), 5) if isinstance(d(), float) else d() for d in distributions]))
            for _ in range(self.n_iter)
        ]