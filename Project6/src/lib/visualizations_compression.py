import numpy as np
import pandas as pd
from IPython.display import display
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from umap import UMAP
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.sparse import csr_matrix
from PIL import Image


def compare_dimensionality_reduction(X, y, models_to_run=None):
    """
    Выполняет полный цикл сравнения методов снижения размерности для датасета.
    Для каждого метода:
    1. Трансформирует данные в 2D.
    2. Строит scatter-plot для визуальной оценки разделения классов.
    3. Обучает k-NN классификатор и оценивает его точность.

    Args:
        X (np.array or pd.DataFrame): Матрица признаков.
        y (np.array or pd.Series): Вектор целевых меток.
        models_to_run (list, optional): Список названий моделей для запуска.
                                        По умолчанию запускаются все.
    """
    
    all_models = {
        'PCA': PCA(n_components=2, random_state=42),
        # TruncatedSVD - аналог PCA для разреженных данных, но хорошо работает и на плотных
        'SVD': TruncatedSVD(n_components=2, random_state=42),
        'Randomized-SVD': TruncatedSVD(n_components=2, algorithm='randomized', random_state=42),
        'TSNE': TSNE(n_components=2, random_state=42, n_jobs=-1, init='pca', learning_rate='auto'),
        'UMAP': UMAP(n_components=2, random_state=42),
        'LLE': LocallyLinearEmbedding(n_components=2, random_state=42, n_jobs=-1, n_neighbors=10)
    }

    # Если пользователь не указал модели, используем все
    if models_to_run is None:
        models_to_run = list(all_models.keys())
    
    # 2. Разделение данных и стандартизация
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Основной цикл по моделям
    results = {}
    for model_name in models_to_run:
        print(f"--- Обработка метода: {model_name} ---")
        
        reducer = all_models[model_name]
        
        # Обучаем трансформер 
        X_train_reduced = reducer.fit_transform(X_train)
        
        # Применяем обученный трансформер к тестовым данным
        # TSNE не имеет стандартного метода transform, поэтому для него fit_transform
        if model_name == 'TSNE':
             X_test_reduced = reducer.fit_transform(X_test)
        else:
             X_test_reduced = reducer.transform(X_test)

        # --- 4. Визуализация результатов на тестовой выборке ---
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=y_test, cmap='tab10', alpha=0.7, s=10)
        plt.title(f'2D проекция MNIST с помощью {model_name}', fontsize=16)
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(y).tolist(), title="Цифры")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
        # --- 5. Оценка качества разделения с помощью k-NN ---
        knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        
        # Обучаем k-NN на сжатых обучающих данных
        knn.fit(X_train_reduced, y_train)
        
        # Оцениваем точность на обучающей и тестовой выборках
        train_accuracy = accuracy_score(y_train, knn.predict(X_train_reduced))
        test_accuracy = accuracy_score(y_test, knn.predict(X_test_reduced))
        
        print(f"Метрика качества разделения для '{model_name}' (k-NN Accuracy):")
        print(f"  Точность на Train: {train_accuracy:.4f}")
        print(f"  Точность на Test:  {test_accuracy:.4f}\n")
        
        results[model_name] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
        
    # Возвращаем DataFrame с итоговыми метриками для удобного сравнения
    return pd.DataFrame(results).T.sort_values(by='test_accuracy', ascending=False)

# --- 1. Функция загрузки и подготовки изображения ---

def load_grayscale_image(image_path):
    """
    Загружает изображение, преобразует его в оттенки серого и возвращает
    в виде 2D numpy-массива с типом float.
    
    Args:
        image_path (str): Путь к файлу изображения.
        
    Returns:
        np.ndarray: 2D-массив, представляющий изображение.
    """
    try:
        # Открываем изображение и конвертируем в 'L' (Luminance) -> оттенки серого
        img = Image.open(image_path).convert('L')
        # Преобразуем в numpy массив с типом float для вычислений
        return np.array(img, dtype=float)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {image_path}")
        return None

# --- 2. Функция для выполнения SVD ---

def decompose_svd(image_matrix):
    """
    Выполняет сингулярное разложение (SVD) матрицы изображения.
    
    Args:
        image_matrix (np.ndarray): 2D-массив изображения.
        
    Returns:
        tuple: (U, s, Vt)
            U (np.ndarray): Левые сингулярные векторы.
            s (np.ndarray): 1D-массив сингулярных значений.
            Vt (np.ndarray): Транспонированные правые сингулярные векторы.
    """
    # full_matrices=False - более эффективный вариант, возвращает матрицы 
    # с меньшими размерами, которых достаточно для реконструкции.
    U, s, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    return U, s, Vt

# --- 3. Функция для восстановления изображения по рангу ---

def reconstruct_image_from_svd(U, s, Vt, rank):
    """
    Восстанавливает (аппроксимирует) изображение, используя заданное 
    количество сингулярных значений (ранг).
    
    Args:
        U, s, Vt: Компоненты SVD.
        rank (int): Количество компонент для использования.
        
    Returns:
        np.ndarray: 2D-массив восстановленного изображения.
    """
    rank = int(rank)
    # Используем broadcasting (U[:, :rank] * s[:rank]) вместо создания
    # полной диагональной матрицы Sigma для эффективности.
    reconstructed_matrix = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    return reconstructed_matrix

# --- 4. "Отчетная" функция для полного анализа и визуализации ---

def plot_svd_analysis(original_image, U, s, Vt, ranks_to_plot):
    """
    Создает и отображает полный набор графиков для анализа SVD:
    - Исходное изображение.
    - Спектр сингулярных значений.
    - График накопленной объясненной дисперсии.
    - Восстановленные изображения для каждого заданного ранга.
    """
    # Создаем сетку для графиков.
    fig, axes = plt.subplots(3, 5, figsize=(16, 12))
    ax_flat = axes.flatten() # Упрощает итерацию по осям

    # --- График 1: Оригинал ---
    ax_flat[0].imshow(original_image, cmap='gray')
    ax_flat[0].set_title("Original")
    ax_flat[0].axis('off')

    # --- График 2: Спектр сингулярных значений ---
    ax_flat[1].plot(range(1, len(s) + 1), s, 'b-')
    ax_flat[1].set_yscale('log') # Лог-шкала обязательна для наглядности
    ax_flat[1].set_title("Singular Value Spectrum (log)")
    ax_flat[1].set_xlabel("Rank")
    ax_flat[1].set_ylabel("Singular Value")
    ax_flat[1].grid(True, which="both", ls="--")

    # --- График 3: Накопленная объясненная дисперсия ---
    # Дисперсия пропорциональна квадрату сингулярных значений
    variance_explained = np.cumsum(s**2) / np.sum(s**2)
    ax_flat[2].plot(range(1, len(s) + 1), variance_explained, 'r-')
    ax_flat[2].set_title("Explained Variance")
    ax_flat[2].set_xlabel("Rank")
    ax_flat[2].set_ylabel("Cumulative Variance (%)")
    ax_flat[2].axhline(y=0.9, linestyle='--', color='g', label='90% Variance')
    ax_flat[2].axhline(y=0.95, linestyle='--', color='orange', label='95% Variance')
    ax_flat[2].legend()
    ax_flat[2].grid(True)
    ax_flat[2].set_ylim(0, 1.05)
    
    # Скрываем пустую  в первом ряду
    ax_flat[3].axis('off')
    ax_flat[4].axis('off')

    # --- Графики 4-12: Восстановленные изображения ---
    # Начинаем с 5-го subplot'а (индекс 4)
    for i, rank in enumerate(ranks_to_plot):
        if i + 4 >= len(ax_flat):
            break # Если рангов больше, чем ячеек, останавливаемся
            
        ax = ax_flat[i + 5]
        reconstructed_img = reconstruct_image_from_svd(U, s, Vt, rank)
        ax.imshow(reconstructed_img, cmap='gray')
        ax.set_title(f"Rank = {rank}")
        ax.axis('off')

    # Скрываем оставшиеся неиспользованные оси
    for j in range(i + 5, len(ax_flat)):
        ax_flat[j].axis('off')

    plt.tight_layout()
    plt.show()