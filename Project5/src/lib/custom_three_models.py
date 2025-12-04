import numpy as np
from collections import Counter
import pandas as pd

class Node:
    """Узел с поддержкой predict_proba."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, y_leaf=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.y_leaf = y_leaf
        self.is_leaf_node = value is not None

class DecisionTree:
    """Базовый класс для алгоритмов решающих деревьев."""
    def __init__(self, max_depth=100, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        if y.ndim > 1:
            y = y.ravel()
        
        self.classes_ = np.sort(np.unique(y))
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth or 
            self._is_finished(y) or 
            n_samples < self.min_samples_split):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value, y_leaf=y.copy())

        # Выбор случайного подмножества признаков (max_features)
        if self.max_features is not None:
            n_features_to_use = min(self.max_features, n_features)
        else:
            n_features_to_use = n_features
        
        feat_idxs = np.random.choice(n_features, n_features_to_use, replace=False)
        best_feat, best_thresh = self._find_split(X, y, feat_idxs)
        
        if best_feat is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value, y_leaf=y.copy())
            
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _find_split(self, X, y, feat_idxs):
        raise NotImplementedError

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node:
            return node.value
        
        if hasattr(x, 'values'):
            x = x.values
        elif isinstance(x, list):
            x = np.array(x)
        
        feature_idx = int(node.feature)
        
        if x[feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _is_finished(self, y):
        raise NotImplementedError

    def _leaf_value(self, y):
        raise NotImplementedError


class CustomDecisionTreeClassifier(DecisionTree):
    """Классификатор решающего дерева с поддержкой predict_proba и max_features."""
    
    def _is_finished(self, y):
        return len(np.unique(y)) == 1

    def _leaf_value(self, y):
        return Counter(y).most_common(1)[0][0]

    def _find_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feat, best_thresh = None, None
        
        parent_gini = self._gini(y)
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            n_samples = len(y)
            
            sort_idxs = np.argsort(X_column)
            X_sorted = X_column[sort_idxs]
            y_sorted = y[sort_idxs]
            
            left_counts = {}
            right_counts = {}
            
            for label in y_sorted:
                right_counts[label] = right_counts.get(label, 0) + 1
            
            for i in range(1, n_samples):
                label = y_sorted[i - 1]
                left_counts[label] = left_counts.get(label, 0) + 1
                right_counts[label] -= 1
                
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                
                n_left = i
                n_right = n_samples - i
                
                gini_left = 1.0 - sum((c / n_left) ** 2 for c in left_counts.values())
                gini_right = 1.0 - sum((c / n_right) ** 2 for c in right_counts.values() if c > 0)
                
                weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0
        
        return best_feat, best_thresh

    def _gini(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def predict_proba(self, X):
        if self.root is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if hasattr(X, 'values'):
            X = X.values
        elif isinstance(X, list):
            try:
                X = np.array(X)
            except (ValueError, TypeError):
                X = np.array([x.values if hasattr(x, 'values') else x for x in X])
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        proba = np.array([self._predict_proba_single(x, self.root) for x in X])
        return proba

    def _predict_proba_single(self, x, node):
        if node.is_leaf_node:
            proba = np.zeros(len(self.classes_))
            
            for i, cls in enumerate(self.classes_):
                count = np.sum(node.y_leaf == cls)
                proba[i] = count / len(node.y_leaf)
            
            return proba
        
        if hasattr(x, 'values'):
            x_array = x.values
        elif isinstance(x, list):
            x_array = np.array(x)
        else:
            x_array = x
        
        feature_idx = int(node.feature)
        
        if x_array[feature_idx] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        return self._predict_proba_single(x, node.right)


class DecisionTreeRegressor(DecisionTree):
    """Регрессор решающего дерева с оптимизированным поиском сплита."""
    
    def _is_finished(self, y):
        return len(np.unique(y)) == 1
        
    def _leaf_value(self, y):
        return np.mean(y)

    def _find_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feat, best_thresh = None, None
        
        parent_var = np.var(y) if len(y) > 0 else 0
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            n_samples = len(y)
            
            sort_idxs = np.argsort(X_column)
            X_sorted = X_column[sort_idxs]
            y_sorted = y[sort_idxs]
            
            left_sum = 0.0
            left_sum_sq = 0.0
            right_sum = np.sum(y_sorted)
            right_sum_sq = np.sum(y_sorted ** 2)
            
            for i in range(1, n_samples):
                val = y_sorted[i - 1]
                left_sum += val
                left_sum_sq += val * val
                right_sum -= val
                right_sum_sq -= val * val
                
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                
                n_left = i
                n_right = n_samples - i
                
                var_left = (left_sum_sq - (left_sum ** 2) / n_left) / n_left if n_left > 0 else 0
                var_right = (right_sum_sq - (right_sum ** 2) / n_right) / n_right if n_right > 0 else 0
                
                weighted_var = (n_left / n_samples) * var_left + (n_right / n_samples) * var_right
                gain = parent_var - weighted_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0
        
        return best_feat, best_thresh


class CustomExtraTreesClassifier(CustomDecisionTreeClassifier):
    """Классификатор экстремально случайных деревьев."""
    
    def _find_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        parent_gini = self._gini(y)
        n_samples = len(y)
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            min_val, max_val = X_column.min(), X_column.max()
            
            if min_val == max_val:
                continue
            
            threshold = np.random.uniform(min_val, max_val)
            
            left_idxs = X_column <= threshold
            right_idxs = ~left_idxs
            
            n_left = np.sum(left_idxs)
            n_right = np.sum(right_idxs)
            
            if n_left == 0 or n_right == 0:
                continue
            
            gini_left = self._gini(y[left_idxs])
            gini_right = self._gini(y[right_idxs])
            
            weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
            gain = parent_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_thresh = threshold
        
        return split_idx, split_thresh


class CustomRandomForestClassifier:
    """
    Классификатор случайного леса с поддержкой predict_proba и max_features.
    
    Использует бэггинг (bootstrap aggregating) с множественными решающими деревьями.
    Поддерживает фиксированный random seed для воспроизводимости.
    
    Атрибуты:
        n_trees (int): Количество деревьев в лесу.
        max_depth (int): Максимальная глубина каждого дерева.
        min_samples_split (int): Минимальное количество образцов для разделения узла.
        max_features (int or str): Количество признаков для рассмотрения при каждом сплите.
            - int: конкретное число признаков
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - None: все признаки (по умолчанию)
        random_state (int): Случайное зерно для воспроизводимости.
        trees (list): Список обученных решающих деревьев.
        classes_ (ndarray): Уникальные метки классов.
    """
    
    def __init__(self, n_trees=10, max_depth=15, min_samples_split=2, max_features=None, random_state=None):
        """
        Инициализация классификатора случайного леса.
        
        Параметры:
            n_trees (int): Количество решающих деревьев для обучения (по умолчанию: 10).
            max_depth (int): Максимальная глубина для каждого дерева (по умолчанию: 15).
            min_samples_split (int): Минимальное количество образцов для разделения (по умолчанию: 2).
            max_features (int or str): Количество признаков на сплит (по умолчанию: None - все признаки).
                Варианты: int, 'sqrt', 'log2', None
            random_state (int): Зерно для воспроизводимости (по умолчанию: None).
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_max_features(self, n_features):
        """
        Вычисление количества признаков для использования.
        
        Параметры:
            n_features (int): Общее количество признаков.
        
        Возвращает:
            int: Количество признаков для использования.
        """
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        """
        Обучение случайного леса на bootstrap-выборках.
        
        Параметры:
            X (array-like): Матрица признаков формы (n_samples, n_features).
            y (array-like): Целевые значения формы (n_samples,).
        
        Возвращает:
            self
        """
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        if y.ndim > 1:
            y = y.ravel()
        
        self.classes_ = np.sort(np.unique(y))
        
        n_samples, n_features = X.shape
        max_features_to_use = self._get_max_features(n_features)
        self.trees = []
        
        for tree_idx in range(self.n_trees):
            # Bootstrap-выборка: n_samples образцов с возвращением
            bootstrap_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_idxs]
            y_bootstrap = y[bootstrap_idxs]
            
            # Каждое дерево получает уникальное зерно на основе random_state и tree_idx
            tree_seed = None
            if self.random_state is not None:
                tree_seed = self.random_state + tree_idx
            
            tree = CustomDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features_to_use,
                random_state=tree_seed
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Предсказание меток классов для образцов в X с использованием мажоритарного голосования.
        
        Параметры:
            X (array-like): Матрица признаков формы (n_samples, n_features).
        
        Возвращает:
            predictions (ndarray): Предсказанные метки классов формы (n_samples,).
        """
        if len(self.trees) == 0:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Получаем предсказания от всех деревьев
        predictions = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        
        # Мажоритарное голосование для каждого образца
        final_predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            votes = predictions[:, i]
            final_predictions[i] = Counter(votes).most_common(1)[0][0]
        
        return final_predictions
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей классов путем усреднения вероятностей от всех деревьев.
        
        Параметры:
            X (array-like): Матрица признаков формы (n_samples, n_features).
        
        Возвращает:
            probabilities (ndarray): Вероятности классов формы (n_samples, n_classes).
                probabilities[i, j] = средняя вероятность класса j для образца i.
        """
        if len(self.trees) == 0:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if hasattr(X, 'values'):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        if hasattr(X, 'ndim') and X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = len(X)
        n_classes = len(self.classes_)
        
        # Усреднение вероятностей от всех деревьев
        avg_proba = np.zeros((n_samples, n_classes))
        
        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            avg_proba += tree_proba
        
        avg_proba /= self.n_trees
        
        return avg_proba
    
    def score(self, X, y):
        """
        Вычисление точности (accuracy).
        
        Параметры:
            X (array-like): Матрица признаков.
            y (array-like): Истинные метки.
        
        Возвращает:
            accuracy (float): Оценка точности.
        """
        predictions = self.predict(X)
        if hasattr(y, 'values'):
            y = y.values
        accuracy = np.mean(predictions == y)
        return accuracy


class GBDTDecisionTreeRegressor(DecisionTree):
    """
    Регрессор решающего дерева для GBDT с поддержкой max_features.
    Используется для обучения на градиентах (остатках).
    """
    
    def __init__(self, max_depth=3, min_samples_split=2, max_features=None, random_state=None):
        super().__init__(max_depth, min_samples_split, max_features, random_state)
    
    def _is_finished(self, y):
        return len(np.unique(y)) == 1
        
    def _leaf_value(self, y):
        return np.mean(y)

    def _find_split(self, X, y, feat_idxs):
        """Оптимизированный поиск сплита для регрессии."""
        best_gain = -1
        best_feat, best_thresh = None, None
        
        parent_var = np.var(y) if len(y) > 0 else 0
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            n_samples = len(y)
            
            # Сортировка один раз на признак
            sort_idxs = np.argsort(X_column)
            X_sorted = X_column[sort_idxs]
            y_sorted = y[sort_idxs]
            
            # Инкрементальное отслеживание сумм
            left_sum = 0.0
            left_sum_sq = 0.0
            right_sum = np.sum(y_sorted)
            right_sum_sq = np.sum(y_sorted ** 2)
            
            for i in range(1, n_samples):
                val = y_sorted[i - 1]
                left_sum += val
                left_sum_sq += val * val
                right_sum -= val
                right_sum_sq -= val * val
                
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                
                n_left = i
                n_right = n_samples - i
                
                # Быстрое вычисление дисперсии
                var_left = (left_sum_sq - (left_sum ** 2) / n_left) / n_left if n_left > 0 else 0
                var_right = (right_sum_sq - (right_sum ** 2) / n_right) / n_right if n_right > 0 else 0
                
                weighted_var = (n_left / n_samples) * var_left + (n_right / n_samples) * var_right
                gain = parent_var - weighted_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0
        
        return best_feat, best_thresh


class CustomGBDTClassifier:
    """
    Классификатор на основе градиентного бустинга решающих деревьев.
    
    Использует бинарную кросс-энтропийную функцию потерь и обучает деревья последовательно на градиентах.
    Реализует инкрементальное обучение, где каждое дерево обучается на остатках
    (градиентах) от предыдущих деревьев.
    
    Атрибуты:
        number_of_trees (int): Количество итераций бустинга.
        max_depth (int): Максимальная глубина каждого дерева.
        max_features (int): Количество признаков для рассмотрения при каждом сплите.
        learning_rate (float): Параметр сжатия (shrinkage).
        min_samples_split (int): Минимальное количество образцов для разделения узла.
        random_state (int): Случайное зерно для воспроизводимости.
        trees (list): Список обученных регрессионных деревьев.
        initial_prediction (float): Начальное предсказание (log-odds).
        classes_ (ndarray): Уникальные метки классов.
    """
    
    def __init__(self, number_of_trees=100, max_depth=3, max_features=None, 
                 learning_rate=0.1, min_samples_split=2, random_state=None):
        """
        Инициализация классификатора GBDT.
        
        Параметры:
            number_of_trees (int): Количество итераций бустинга (по умолчанию: 100).
            max_depth (int): Максимальная глубина для каждого дерева (по умолчанию: 3).
            max_features (int): Количество признаков на сплит (None = все признаки).
            learning_rate (float): Скорость обучения для градиентного спуска (по умолчанию: 0.1).
            min_samples_split (int): Минимальное количество образцов для разделения (по умолчанию: 2).
            random_state (int): Случайное зерно для воспроизводимости.
        """
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.trees = []
        self.initial_prediction = None
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, z):
        """
        Функция активации sigmoid.
        """
        z = np.clip(z, -500, 500)  # Предотвращение переполнения
        return 1.0 / (1.0 + np.exp(-z))
    
    def _binary_cross_entropy_gradient(self, y_true, y_pred_proba):
        """
        Вычисление градиента бинарной кросс-энтропийной функции потерь.
        
        Параметры:
            y_true (ndarray): Истинные метки (0 или 1).
            y_pred_proba (ndarray): Предсказанные вероятности.
        
        Возвращает:
            gradients (ndarray): Отрицательный градиент (остатки).
        """
        return y_true - y_pred_proba
    
    def fit(self, X, y):
        """
        Обучение классификатора GBDT с использованием градиентного бустинга.
                
        Параметры:
            X (array-like): Матрица признаков формы (n_samples, n_features).
            y (array-like): Бинарные целевые значения (0 или 1).
        
        Возвращает:
            self
        """
        # Преобразование в numpy массивы
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        if y.ndim > 1:
            y = y.ravel()
        
        # Сохранение классов
        self.classes_ = np.sort(np.unique(y))
        
        if len(self.classes_) != 2:
            raise ValueError("CustomGBDTClassifier only supports binary classification (2 classes)")
        
        n_samples = len(X)
        
        # Шаг 1: Инициализация с log-odds
        # F_0(x) = log(p / (1-p)), где p = доля положительного класса
        p_positive = np.mean(y)
        if p_positive == 0:
            p_positive = 1e-15
        elif p_positive == 1:
            p_positive = 1 - 1e-15
        
        self.initial_prediction = np.log(p_positive / (1 - p_positive))
        
        # Инициализация предсказаний (сырые скоры, не вероятности)
        F = np.full(n_samples, self.initial_prediction)
        
        # Шаг 2: Итеративное обучение деревьев на градиентах (инкрементальное обучение)
        for tree_idx in range(self.number_of_trees):
            # Преобразование сырых скоров в вероятности с использованием sigmoid
            y_pred_proba = self._sigmoid(F)
            
            # Вычисление градиентов (отрицательный градиент функции потерь = остатки)
            gradients = self._binary_cross_entropy_gradient(y, y_pred_proba)
            
            # Обучение регрессионного дерева на градиентах с уникальным random_state
            tree_seed = None
            if self.random_state is not None:
                tree_seed = self.random_state + tree_idx
            
            tree = GBDTDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X, gradients)
            
            # Получение предсказаний от этого дерева
            tree_predictions = tree.predict(X)
            
            # Обновление F со скоростью обучения (инкрементальное обновление)
            F += self.learning_rate * tree_predictions
            
            # Сохранение дерева
            self.trees.append(tree)
        
        return self
    
    def _predict_raw(self, X):
        """
        Предсказание сырых скоров (до sigmoid).
        
        Возвращает:
            raw_scores (ndarray): Сырые предсказания (log-odds).
        """
        if hasattr(X, 'values'):
            X = X.values
        
        n_samples = len(X)
        
        # Начало с начального предсказания
        F = np.full(n_samples, self.initial_prediction)
        
        # Добавление вкладов от всех деревьев (инкрементальные предсказания)
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            F += self.learning_rate * tree_predictions
        
        return F
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей классов.
        
        Параметры:
            X (array-like): Матрица признаков.
        
        Возвращает:
            probabilities (ndarray): Форма (n_samples, 2)
                probabilities[:, 0] = P(y=0)
                probabilities[:, 1] = P(y=1)
        """
        if len(self.trees) == 0:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Получение сырых скоров
        raw_scores = self._predict_raw(X)
        
        # Преобразование в вероятности с использованием sigmoid
        proba_positive = self._sigmoid(raw_scores)
        proba_negative = 1 - proba_positive
        
        # Объединение в массив формы (n_samples, 2)
        probabilities = np.column_stack([proba_negative, proba_positive])
        
        return probabilities
    
    def predict(self, X):
        """
        Предсказание меток классов.
        
        Параметры:
            X (array-like): Матрица признаков.
        
        Возвращает:
            predictions (ndarray): Предсказанные метки классов (0 или 1).
        """
        probabilities = self.predict_proba(X)
        # Возврат класса с наивысшей вероятностью
        return np.argmax(probabilities, axis=1)
    
    def score(self, X, y):
        """
        Вычисление оценки точности (accuracy).
        
        Параметры:
            X (array-like): Матрица признаков.
            y (array-like): Истинные метки.
        
        Возвращает:
            accuracy (float): Оценка точности.
        """
        predictions = self.predict(X)
        if hasattr(y, 'values'):
            y = y.values
        accuracy = np.mean(predictions == y)
        return accuracy
