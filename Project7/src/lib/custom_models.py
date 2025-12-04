import numpy as np
from collections import deque
from scipy.stats import multivariate_normal



class CustomKMeans:
    def __init__(self, n_clusters=8, random_state=None, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # случайный выбор начальных центроидов
        initial_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X.iloc[initial_idxs].values
        
        for i in range(self.max_iter):
            # присваиваем метки (ближайший центр)
            labels = np.argmin(
                np.linalg.norm(X.values[:, None] - centers[None, :], axis=2),
                axis=1
            )
            
            # пересчитываем центры
            new_centers = np.array([X.values[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(self.n_clusters)])
            
            # проверка сходимости
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels

        distances = np.linalg.norm(X.values - centers[labels], axis=1)
        self.inertia_ = np.sum(distances ** 2)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

class CustomDBSCAN:
    def __init__(self, eps=0.001, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        if hasattr(X, 'values'):
            X_vals = X.values
        else:
            X_vals = X
        n_samples = X_vals.shape[0]
        labels = np.full(n_samples, -1, dtype=int)  # -1 для шумов
        cluster_id = 0
        
        # Предварительно считаем соседей для каждой точки
        neighbors = [self.region_query(X_vals, i) for i in range(n_samples)]
        
        for i in range(n_samples):
            if labels[i] != -1:
                # Уже обработана точка
                continue
            if len(neighbors[i]) < self.min_samples:
                # Шум или пограничная точка
                labels[i] = -1
                continue
            
            # Начинаем новый кластер
            labels[i] = cluster_id
            queue = deque(neighbors[i])
            
            while queue:
                j = queue.popleft()
                if labels[j] == -1:
                    labels[j] = cluster_id
                if labels[j] != -1:
                    continue
                labels[j] = cluster_id
                
                if len(neighbors[j]) >= self.min_samples:
                    queue.extend(neighbors[j])
            
            cluster_id += 1
        
        self.labels_ = labels
        self.X_vals = X_vals
        return self
    
    def predict(self, X_new):
        # Для каждой точки из X_new ищем ближайший плотный кластер по eps
        if hasattr(X_new, 'values'):
            X_vals = X_new.values
        else:
            X_vals = X_new
        n_samples = X_vals.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        
        # Если модель еще не обучена - ошибка
        if self.labels_ is None:
            raise Exception("Call fit before predict")
        
        # Для более эффективного поиска можно использовать KD-дерево,
        # но здесь простой перебор
        for i in range(n_samples):
            dists = np.linalg.norm(self.X_vals - X_vals[i], axis=1)
            neighbors = np.where(dists <= self.eps)[0]
        
            if len(neighbors) < self.min_samples:
                # Относим к шуму
                labels[i] = -1
            else:
                # Присваиваем кластер самого близкого "ядра"
                # Предполагая первый ближайший сосед
                cluster_labels = self.labels_[neighbors]
                cluster_labels = cluster_labels[cluster_labels != -1]
                if len(cluster_labels) == 0:
                    labels[i] = -1
                else:
                    # Выбираем самый частый метку среди соседей
                    labels[i] = np.bincount(cluster_labels).argmax()
        
        return labels
    
    def region_query(self, X, idx):
        # Возвращает индексы точек в eps радиусе вокруг точки idx
        dists = np.linalg.norm(X - X[idx], axis=1)
        return np.where(dists <= self.eps)[0]
    

class CustomGaussianMixture:
    def __init__(self, n_components=30, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Параметры модели
        self.weights_ = None             # mixing weights
        self.means_ = None               # means of components
        self.covariances_ = None         # covariances of components
        self.converged_ = False
        self.resp_ = None                # responsibilities
        self.n_iter_ = 0

    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Инициализируем равномерные веса
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Инициализируем случайные выборки как начальные центры
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices]
        
        # Инициализируем ковариации как единичные матрицы
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def _e_step(self, X):
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)
            resp[:, k] = self.weights_[k] * mvn.pdf(X)
        
        # нормализуем, чтобы суммы по компонентам равнялись 1
        resp_sum = resp.sum(axis=1)[:, np.newaxis]
        resp_sum[resp_sum == 0] = 1e-15
        resp = resp / resp_sum
        return resp
    
    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)  # эффективные числа точек в кластерах
        
        self.weights_ = Nk / n_samples
        
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
        
        covariances = []
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_cov = np.dot(resp[:, k] * diff.T, diff) / Nk[k]
            covariances.append(weighted_cov + 1e-15 * np.eye(n_features))  # добавим регуляризацию
        self.covariances_ = np.array(covariances)
    
    def fit(self, X):
        if hasattr(X, 'values'):
            X = X.values
        self._initialize_parameters(X)
        
        log_likelihood_old = None
        
        for i in range(self.max_iter):
            self.resp_ = self._e_step(X)
            self._m_step(X, self.resp_)
            
            # Вычисляем log-likelihood
            probs = np.sum([
                self.weights_[k] * multivariate_normal.pdf(X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True) 
                for k in range(self.n_components)
            ], axis=0)

            # Защита от нулей
            probs = np.clip(probs, 1e-15, None)

            log_likelihood = np.sum(np.log(probs))
            
            if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < self.tol:
                self.converged_ = True
                break
            
            log_likelihood_old = log_likelihood
        
        self.n_iter_ = i + 1
        return self
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)
            resp[:, k] = self.weights_[k] * mvn.pdf(X)
        labels = np.argmax(resp, axis=1)
        return labels
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    
    def score(self, X):
        if hasattr(X, 'values'):
            X_vals = X.values
        else:
            X_vals = X
        # Для каждой компоненты считаем плотность * вес
        probs = np.sum([
            self.weights_[k] * multivariate_normal.pdf(X_vals, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)
            for k in range(self.n_components)
        ], axis=0)
        # Чтобы избежать log(0), минимальное значение
        probs = np.clip(probs, 1e-15, None)
        # Возвращаем средний логарифм правдоподобия на точку
        return np.mean(np.log(probs))