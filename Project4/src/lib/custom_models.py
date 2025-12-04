import numpy as np
from collections import Counter


class CustomLogisticRegression:
    """
    Кастомная реализация логистической регрессии с использованием
    стохастического градиентного спуска (SGD).
    """
    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=32, verbose=False):
        """
        Инициализация гиперпараметров модели.

        Args:
            learning_rate (float): Скорость обучения, размер шага градиентного спуска.
            n_epochs (int): Количество эпох (полных проходов по данным) для обучения.
            batch_size (int): Размер батча для стохастического градиентного спуска.
            verbose (bool): Если True, будет выводить лосс каждые 100 эпох.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.w = None  # Веса для признаков
        self.b = None  # Смещение (bias)

    def _sigmoid(self, z):
        """Вспомогательная сигмоидная функция."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Обучает модель, находя оптимальные веса w и b с помощью SGD.
        """
        # Убедимся, что X и y - это numpy массивы
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        
        # 1. Инициализация параметров
        self.w = np.zeros(n_features)
        self.b = 0

        # 2. Основной цикл обучения по эпохам
        for epoch in range(self.n_epochs):
            # 3. Цикл по мини-батчам
            for i in range(0, n_samples, self.batch_size):
                # Выбираем текущий батч
                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]
                
                # --- Расчет градиентов ---
                # Линейная комбинация
                linear_model = np.dot(X_batch, self.w) + self.b
                # Предсказание вероятности
                y_predicted = self._sigmoid(linear_model)

                # Градиенты для весов (dw) и смещения (db)
                dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / self.batch_size) * np.sum(y_predicted - y_batch)
                # -------------------------

                # 4. Шаг градиентного спуска: обновление параметров
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            
            # Вывод лосса для мониторинга (по желанию)
            if self.verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                # Рассчитаем лосс на всем датасете для более стабильной оценки
                total_loss = -np.mean(y * np.log(self._sigmoid(np.dot(X, self.w) + self.b) + 1e-9) + \
                                     (1 - y) * np.log(1 - self._sigmoid(np.dot(X, self.w) + self.b) + 1e-9))
                print(f"Эпоха {epoch}: Лосс = {total_loss:.4f}")

    def predict_proba(self, X):
        """
        Предсказывает вероятности принадлежности к классу '1'.
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Модель должна быть обучена перед предсказанием. Вызовите метод fit().")
        X = np.asarray(X)
        linear_model = np.dot(X, self.w) + self.b
        
        # Возвращаем вероятности для класса '1'
        proba_class_1 = self._sigmoid(linear_model)
        
        # Формируем массив (n_samples, 2) как в sklearn
        return np.column_stack([1 - proba_class_1, proba_class_1])


    def predict(self, X, threshold=0.5):
        """
        Предсказывает метки классов (0 или 1) на основе порога.
        """
        # Получаем вероятности для класса '1'
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)


class CustomKNNClassifier:
    """
    Кастомная реализация классификатора K-ближайших соседей (KNN).
    """
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Инициализация гиперпараметров.

        Args:
            n_neighbors (int): Количество соседей для голосования.
            weights (str): Способ взвешивания соседей. 'uniform' (все равны) 
                           или 'distance' (вес обратно пропорционален расстоянию).
        """
        if weights not in ['uniform', 'distance']:
            raise ValueError("Параметр weights может быть только 'uniform' или 'distance'")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        "Обучение" модели. Для KNN это простое запоминание данных.
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _get_neighbors_and_dists(self, X_test):
        n_test_samples = X_test.shape[0]
        # Создаем пустые массивы для результатов
        k_indices = np.zeros((n_test_samples, self.n_neighbors), dtype=int)
        k_distances = np.zeros((n_test_samples, self.n_neighbors), dtype=float)

        # Итерируемся по каждой тестовой точке
        for i in range(n_test_samples):
            # Вычисляем расстояние от одной тестовой точки до всех точек трейна
            distances = np.sqrt(np.sum((self.X_train - X_test[i])**2, axis=1))
            
            # Находим k ближайших соседей для этой одной точки
            indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            
            k_indices[i] = indices
            k_distances[i] = distances[indices]
            
        return k_indices, k_distances
    
    def predict_proba(self, X):
        """
        Предсказывает вероятности принадлежности к каждому классу.
        """
        if self.X_train is None:
            raise RuntimeError("Модель должна быть обучена перед предсказанием. Вызовите метод fit().")
        
        X = np.asarray(X)
        n_test_samples = X.shape[0]
        
        # Находим соседей
        neighbor_indices, neighbor_dists = self._get_neighbors_and_dists(X)
        
        # Массив для хранения вероятностей
        probas = np.zeros((n_test_samples, 2)) # Для классов 0 и 1

        # Итерируемся по каждой тестовой точке
        for i in range(n_test_samples):
            # Получаем метки соседей
            neighbor_labels = self.y_train[neighbor_indices[i]]

            if self.weights == 'uniform':
                # Считаем количество голосов за каждый класс
                counts = Counter(neighbor_labels)
                proba_class_1 = counts.get(1, 0) / self.n_neighbors
            
            elif self.weights == 'distance':
                # Добавляем небольшое число, чтобы избежать деления на ноль
                dists = neighbor_dists[i] + 1e-9
                # Вес = 1 / расстояние
                weights = 1 / dists

                # Сумма весов для каждого класса
                weighted_sum_class_1 = np.sum(weights[neighbor_labels == 1])
                total_weighted_sum = np.sum(weights)
                
                if total_weighted_sum > 0:
                    proba_class_1 = weighted_sum_class_1 / total_weighted_sum
                else: # Если все расстояния были 0 (точка совпала с соседями)
                    proba_class_1 = np.mean(neighbor_labels)

            probas[i, 1] = proba_class_1
            probas[i, 0] = 1 - proba_class_1
            
        return probas

    def predict(self, X, threshold=0.5):
        """
        Предсказывает метки классов (0 или 1) на основе порога.
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

class CustomGaussianNB:
    """
    Кастомная реализация Гауссова Наивного Байесовского классификатора.
    """
    def __init__(self, var_smoothing=1e-09):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.mean_ = None
        self.var_ = None
        self.class_prior_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            if X_c.shape[0] == 0:
                # Если для класса нет данных, используем средние по всему датасету
                self.mean_[idx, :] = np.nanmean(X, axis=0)
                self.var_[idx, :] = np.nanvar(X, axis=0)
            else:
                self.mean_[idx, :] = np.nanmean(X_c, axis=0)
                self.var_[idx, :] = np.nanvar(X_c, axis=0)

            self.class_prior_[idx] = X_c.shape[0] / float(n_samples)

        # Добавляем сглаживание к дисперсии 
        max_var = np.max(self.var_)
        self.var_ += self.var_smoothing * max_var
        # ---------------------------------

    def _pdf(self, class_idx, x):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        # Теперь `var` гарантированно не ноль, эпсилон не нужен
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.clip(numerator / denominator, self.var_smoothing, None) 

    def predict_proba(self, X):
        """
        Предсказывает вероятности принадлежности к каждому классу.
        """
        if self.mean_ is None:
            raise RuntimeError("Модель должна быть обучена перед предсказанием. Вызовите метод fit().")

        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Массив для хранения постериорных вероятностей
        posteriors = np.zeros((n_samples, len(self.classes_)))

        # Для каждого класса...
        for idx, c in enumerate(self.classes_):
            # ...берем его априорную вероятность
            prior = np.log(self.class_prior_[idx])
            # ...и вычисляем условные вероятности P(x_i|C=c) для всех признаков
            # Суммируем логарифмы вместо перемножения вероятностей, чтобы избежать проблем с очень малыми числами
            conditional_prob = np.sum(np.log(self._pdf(idx, X)), axis=1)
            
            # Постериорная вероятность P(C=c|X) пропорциональна P(C=c) * P(X|C=c)
            posteriors[:, idx] = prior + conditional_prob
            
        # Нормализуем, чтобы получить реальные вероятности (сумма = 1)
        posteriors_exp = np.exp(posteriors - np.max(posteriors, axis=1, keepdims=True))
        probas = posteriors_exp / np.sum(posteriors_exp, axis=1, keepdims=True)

        return probas
        
    def predict(self, X, threshold=0.5):
        """
        Предсказывает метки классов (0 или 1) на основе порога.
        """
        # Получаем вероятности
        probabilities = self.predict_proba(X)
        
        if probabilities.shape[1] == 2:
            return (probabilities[:, 1] >= threshold).astype(int)
        else: # Для мультиклассовой классификации
            return self.classes_[np.argmax(probabilities, axis=1)]