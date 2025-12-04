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



def search_params(X_train, y_train, X_valid, y_valid, estimator, params, n_iter=20, cv=5, num_top_models = 3):
    """
    Выполняет байесовский поиск гиперпараметров, оценивает топ-3 модели на
    валидационной выборке и возвращает параметры лучшей из них.

    Args:
        X_train: Обучающие признаки.
        y_train: Обучающая целевая переменная.
        X_valid: Валидационные признаки.
        y_valid: Валидационная целевая переменная.
        estimator: Модель scikit-learn (например, Ridge()).
        params: Пространство поиска гиперпараметров для BayesSearchCV.
        n_iter (int): Количество итераций байесовского поиска.
        cv (int): Количество фолдов для кросс-валидации.

    Returns:
        tuple: Кортеж из трех элементов:
               1. best_params (dict): Словарь с параметрами лучшей модели.
               2. best_cv_mse (float): MSE этой модели на кросс-валидации.
               3. best_valid_mse (float): MSE этой модели на валидационной выборке.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=params,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0 
        )

        # Запускаем поиск на обучающих данных
        bayes_search.fit(X_train, y_train)

    print("Поиск завершен. Оценка топ-3 моделей на валидационной выборке...")

    # Получаем результаты всех итераций
    cv_results = bayes_search.cv_results_
    top_models_results = []
    
    # Определяем, сколько моделей проверять (не больше 3 и не больше, чем было итераций)
    num_top_models = min(num_top_models, n_iter)

    for rank in range(1, num_top_models + 1):
        # Находим индекс модели с соответствующим рангом
        model_index = np.where(cv_results['rank_test_score'] == rank)[0][0]

        # Получаем параметры и CV-оценку
        model_params = cv_results['params'][model_index]
        cv_mse = -cv_results['mean_test_score'][model_index]

        # Создаем, обучаем и оцениваем модель с этими параметрами
        model = estimator.set_params(**model_params)
        model.fit(X_train, y_train)
        y_pred_valid = model.predict(X_valid)
        valid_mse = mean_squared_error(y_valid, y_pred_valid)

        print(f"  Ранг {rank} (CV): CV MSE={cv_mse:.4f}, Valid MSE={valid_mse:.4f}, Параметры: {model_params}")
        
        top_models_results.append({
            'params': model_params,
            'cv_mse': cv_mse,
            'valid_mse': valid_mse
        })

    # Находим лучшую модель по результатам на ВАЛИДАЦИОННОЙ выборке
    best_model_on_validation = min(top_models_results, key=lambda x: x['valid_mse'])
    
    best_params = best_model_on_validation['params']
    best_cv_mse = best_model_on_validation['cv_mse']
    best_valid_mse = best_model_on_validation['valid_mse']

    print("\n--- Лучшая модель (по оценке на валидации) ---")
    print(f"  Параметры: {best_params}")
    print(f"  CV MSE: {best_cv_mse:.4f}")
    print(f"  Validation MSE: {best_valid_mse:.4f}\n")

    return best_params, best_cv_mse, best_valid_mse

def evaluate_models(X_train, y_train, X_valid, y_valid, models, 
                    X_test=None, y_test=None,
                    scoring_func=mean_squared_error):
    """
    Обучает, измеряет время и оценивает список моделей на обучающей, 
    валидационной и опционально на тестовой выборках.

    Args:
        X_train, y_train: Обучающие данные.
        X_valid, y_valid: Валидационные данные.
        models (list): Список экземпляров моделей для оценки.
        X_test (optional), y_test (optional): Тестовые данные.
        scoring_func (callable): Функция для оценки (по умолчанию MSE).

    Returns:
        pd.DataFrame: DataFrame с результатами оценки.
    """
    model_names = [type(m).__name__ for m in models]
    
    # Определяем строки в итоговой таблице
    index_rows = ['Fit Time (s)', 'Train', 'Validation']
    if X_test is not None and y_test is not None:
        index_rows.append('Test')
        
    results_df = pd.DataFrame(index=index_rows, columns=model_names)

    # Основной цикл по всем моделям
    for model in models:
        model_name = type(model).__name__
        print(f"--- Оценка модели: {model_name} ---")

        try:
            # 1. Обучение и замер времени
            start_time = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start_time
            results_df.loc['Fit Time (s)', model_name] = fit_time

            # 2. Оценка на обучающей выборке
            train_pred = model.predict(X_train)
            train_score = scoring_func(y_train, train_pred)
            results_df.loc['Train', model_name] = train_score

            # 3. Оценка на валидационной выборке
            val_pred = model.predict(X_valid)
            val_score = scoring_func(y_valid, val_pred)
            results_df.loc['Validation', model_name] = val_score

            # 4. Опциональная оценка на тестовой выборке
            if X_test is not None and y_test is not None:
                test_pred = model.predict(X_test)
                test_score = scoring_func(y_test, test_pred)
                results_df.loc['Test', model_name] = test_score
        
        except Exception as e:
            print(f"    ! Ошибка при оценке модели {model_name}: {e}")
            for row in index_rows:
                 results_df.loc[row, model_name] = 'Error'

    print(f"\n--- Сводная таблица результатов (метрика: {scoring_func.__name__}) ---")
    display(results_df.apply(pd.to_numeric, errors='coerce').round(3))

def select_features_by_score(X_train, y_train, X_valid, X_test, score_threshold=0.0):
    """
    Автоматически отбирает признаки, оценка которых выше заданного порога.

    Args:
        X_train, y_train: Обучающие данные.
        X_valid, X_test: Валидационные и тестовые данные для трансформации.
        score_threshold (float): Порог для F-статистики. Признаки с оценкой
                                 ниже или равной этому порогу будут отброшены.
                                 По умолчанию отбрасываются только признаки с нулевой оценкой.

    Returns:
        tuple: (X_train_selected, X_valid_selected, X_test_selected, final_selector)
               Трансформированные наборы данных и финальный, настроенный селектор.
    """
    print("--- Шаг 1: Оценка всех признаков ---")
    # Используем k='all', чтобы получить оценки для всех признаков
    initial_selector = SelectKBest(score_func=f_regression, k='all')
    initial_selector.fit(X_train, y_train)
    
    all_scores = initial_selector.scores_
    
    # --- Шаг 2: Анализ оценок ---
    
    # Количество признаков с оценкой строго выше нуля
    non_zero_scores_count = np.sum(all_scores > 0)
    print(f"Всего признаков с ненулевой оценкой: {non_zero_scores_count}")
    
    # Количество признаков, прошедших пользовательский порог
    k_best = np.sum(all_scores > score_threshold)
    
    if k_best == 0:
        print(f"Внимание: Ни один признак не прошел порог {score_threshold}. Увеличьте k до 1, чтобы избежать ошибки.")
        k_best = 1 # Устанавливаем k=1, чтобы избежать падения

    print(f"Признаков с оценкой > {score_threshold}: {k_best}")

    # --- Шаг 3: Финальный отбор и трансформация ---
    print(f"\n--- Шаг 3: Создание финального селектора с k={k_best} и трансформация данных ---")
    final_selector = SelectKBest(score_func=f_regression, k=k_best)
    
    # Обучаем финальный селектор. fit_transform для обучающей выборки
    X_train_selected = final_selector.fit_transform(X_train, y_train)
    
    # Просто .transform для валидационной и тестовой
    X_valid_selected = final_selector.transform(X_valid)
    X_test_selected = final_selector.transform(X_test)
    
    print("Трансформация завершена.")
    print(f"Новая форма X_train: {X_train_selected.shape}")
    
    return X_train_selected, X_valid_selected, X_test_selected


def evaluate_pca_components(X_train_sparse, n_components_range, variance_threshold=0.95):
    """
    Оценивает необходимое количество компонент для TruncatedSVD (аналог PCA).
    Строит график объясненной и накопленной дисперсии.

    Args:
        X_train_sparse: Обучающие признаки в разреженном формате.
        n_components_range (range or list): Диапазон количества компонент для проверки.
        variance_threshold (float): Порог объясненной дисперсии для отметки на графике.
    """
    # Обучаем SVD один раз на максимальном числе компонент для эффективности
    max_components = max(n_components_range)
    svd = TruncatedSVD(n_components=max_components, random_state=42, algorithm='randomized')
    svd.fit(X_train_sparse)
    
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    indices_above_threshold = np.where(cumulative_variance >= variance_threshold)[0]
    
    if len(indices_above_threshold) > 0:
        # Если есть такие индексы, берем первый из них (+1 для перевода в количество)
        n_for_threshold = indices_above_threshold[0] + 1
        print(f"Для достижения {variance_threshold:.0%} объясненной дисперсии требуется: {n_for_threshold} компонент.")
    else:
        # Если таких индексов нет, порог не достигнут
        n_for_threshold = -1 # Используем -1 как флаг "не найдено"
        print(f"Порог в {variance_threshold:.0%} не был достигнут с {max_components} компонентами.")
        print(f"Максимальная достигнутая дисперсия: {cumulative_variance[-1]:.2%}")

    # --- Построение графиков ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    components_to_plot = range(1, max_components + 1)

    # График 1: Объясненная дисперсия (Каменная осыпь)
    ax1.bar(components_to_plot, explained_variance, alpha=0.7, label='Индивидуальная дисперсия')
    ax1.set_title('График объясненной дисперсии (Scree Plot)')
    ax1.set_xlabel('Главные компоненты')
    ax1.set_ylabel('Доля объясненной дисперсии')
    ax1.legend()
    ax1.grid(True)

    # График 2: Накопленная дисперсия
    ax2.plot(components_to_plot, cumulative_variance, 'r-o', label='Накопленная дисперсия')
    ax2.axhline(y=variance_threshold, color='g', linestyle='--', label=f'Порог {variance_threshold:.0%}')
    if n_for_threshold > 0:
        ax2.axvline(x=n_for_threshold, color='purple', linestyle=':', label=f'{n_for_threshold} компонент')
    ax2.set_title('График накопленной дисперсии')
    ax2.set_xlabel('Количество компонент')
    ax2.set_ylabel('Накопленная доля дисперсии')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Анализ главных компонент (TruncatedSVD)', fontsize=16)
    plt.show()

    return n_for_threshold

def evaluate_umap_components(X_train_sparse, y_train, X_valid, y_valid, n_components_range):
    """
    Оценивает оптимальное количество компонент для UMAP, измеряя качество
    downstream-модели на валидационной выборке.

    Args:
        X_train_sparse, y_train: Обучающие данные.
        X_valid, y_valid: Валидационные данные.
        n_components_range (range or list): Диапазон количества компонент для проверки.
    """
    valid_scores = []
    
    print("Оценка оптимального числа компонент для UMAP...")
    for n in n_components_range:
        print(f"  Тестирование с n_components = {n}...")
        
        # 1. Обучаем UMAP и преобразуем данные
        umap_reducer = UMAP(n_components=n, n_jobs=-1, low_memory=True)
        X_train_reduced = umap_reducer.fit_transform(X_train_sparse)
        X_valid_reduced = umap_reducer.transform(X_valid)
        
        # 2. Обучаем быструю модель на новых признаках
        model = LinearRegression()
        model.fit(X_train_reduced, y_train)
        
        # 3. Считаем ошибку на валидации
        y_pred = model.predict(X_valid_reduced)
        mse = mean_squared_error(y_valid, y_pred)
        valid_scores.append(mse)

    # Находим оптимальное n по минимальной ошибке
    best_n_index = np.argmin(valid_scores)
    best_n = list(n_components_range)[best_n_index]
    
    # --- Построение графика ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, valid_scores, 'b-o', label='Validation MSE')
    plt.axvline(x=best_n, color='g', linestyle='--', label=f'Оптимум n_components = {best_n}')
    plt.title('Качество модели (MSE) в зависимости от числа компонент UMAP')
    plt.xlabel('Количество компонент UMAP')
    plt.ylabel('Mean Squared Error на валидации')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_n


def transform_features(X_train, X_valid, X_test, index, n_components, method):
    """
    Применяет выбранный метод снижения размерности (TruncatedSVD или UMAP)
    к наборам данных.

    Важно: модель обучается только на X_train, а затем трансформирует все наборы.

    Args:
        X_train, X_valid, X_test: Исходные наборы признаков (разреженные матрицы).
        n_components (int): Целевое количество компонент.
        method (str): Метод для использования. Должен быть 'TruncatedSVD' или 'umap'.

    Returns:
        tuple: Кортеж из трех pandas DataFrame:
               (X_train_reduced_df, X_valid_reduced_df, X_test_reduced_df)
    """
    if method not in ['TruncatedSVD', 'umap']:
        raise ValueError("Метод должен быть 'TruncatedSVD' или 'umap'")

    print(f"--- Применение метода: {method} с {n_components} компонентами ---")

    # 1. Инициализация модели
    if method == 'TruncatedSVD':
        reducer = TruncatedSVD(n_components=n_components, random_state=42, algorithm='randomized')
    elif method == 'umap':
        #
        reducer = UMAP(n_components=n_components, n_jobs=-1, low_memory=True)
    
    # 2. Обучение модели ТОЛЬКО на обучающих данных
    print("Обучение модели на X_train...")
    reducer.fit(X_train)

    # 3. Трансформация всех наборов данных с помощью ОБУЧЕННОЙ модели
    print("Трансформация X_train, X_valid и X_test...")
    X_train_reduced = reducer.transform(X_train)
    X_valid_reduced = reducer.transform(X_valid)
    X_test_reduced = reducer.transform(X_test)
    print("Трансформация завершена.")

    # 4. Создание DataFrame с именованными столбцами
    new_columns = [f'component_{i+1}' for i in range(n_components)]
    
    X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=new_columns, index=index[0])
    X_valid_reduced_df = pd.DataFrame(X_valid_reduced, columns=new_columns, index=index[1])
    X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=new_columns, index=index[2])
    
    return X_train_reduced_df, X_valid_reduced_df, X_test_reduced_df

def calculate_sparsity(df, df_name, epsilon=1e-9):
    """
    Рассчитывает и выводит процент разреженности (долю нулей) в DataFrame.
    
    Args:
        df (pd.DataFrame): Входной DataFrame.
        epsilon (float): Порог, ниже которого значение считается нулем.
                         По умолчанию очень маленькое число, чтобы ловить только "почти нули".
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Входные данные должны быть pandas DataFrame.")
        
    # Считаем количество ячеек, значения в которых по модулю меньше epsilon
    num_zeros = np.sum(np.abs(df.values) < epsilon)
    
    # Общее количество ячеек в DataFrame
    total_elements = df.shape[0] * df.shape[1]
    
    # Рассчитываем долю нулей (разреженность)
    sparsity_percentage = 100 * (num_zeros / total_elements)
    
    print(f"--- Анализ разреженности {df_name} ---")
    print(f"Форма DataFrame: {df.shape}")
    print(f"Всего элементов: {total_elements:,}")
    print(f"Элементов, близких к нулю (абс. значение < {epsilon}): {num_zeros:,}")
    print(f"Процент разреженности (доля нулей): {sparsity_percentage:.2f}%")
    
    if sparsity_percentage > 80:
        print("\nВЫВОД: Высокая разреженность. Преобразование в 'sparse' формат имеет смысл.")
    elif sparsity_percentage > 50:
        print("\nВЫВОД: Умеренная разреженность. Преобразование в 'sparse' может дать небольшой выигрыш.")
    else:
        print("\nВЫВОД: Низкая разреженность. DataFrame является плотным. Преобразование в 'sparse' не рекомендуется.")

def df_to_sparse(df):

    epsilon = 1e-6
    df[np.abs(df) < epsilon] = 0
    numpy_array = df.values
    sparse_matrix = csr_matrix(numpy_array)

    return sparse_matrix

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