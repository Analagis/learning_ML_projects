import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, precision_score, f1_score, average_precision_score
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from numpy.linalg import matrix_rank


def gini_score_sklearn(y_true, y_prob) -> float:
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1

def evaluate_models(train_df, val_df, target_column, models, 
                    scoring_func=gini_score_sklearn, 
                    feature_pools=None):
    """
    Обучает и оценивает список моделей на полном наборе признаков и на заданных поднаборах (пулах).

    Args:
        train_df (pd.DataFrame): Полный тренировочный датафрейм (с целевой колонкой).
        val_df (pd.DataFrame): Полный валидационный датафрейм (с целевой колонкой).
        target_column (str): Название целевой колонки.
        models (list): Список экземпляров моделей для оценки.
        scoring_func (callable): Функция для оценки, принимающая (y_true, y_prob).
        feature_pools (dict, optional): Словарь вида {название_пула: список_признаков}.
                                       Позволяет оценить модели на разных наборах признаков.

    Returns:
        dict: Словарь, где ключ - название пула признаков, а значение - DataFrame с результатами.
    """
    # 1. Разделение данных на полный набор признаков (X) и цель (y)
    X_train_full = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val_full = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    # 2. Формирование словаря с наборами признаков для оценки
    pools_to_evaluate = {'Full Features': X_train_full.columns.tolist()}
    if feature_pools:
        # Добавляем пользовательские наборы признаков
        pools_to_evaluate.update(feature_pools)

    all_results = {} # Словарь для хранения всех таблиц с результатами

    # 3. Основной цикл по всем наборам признаков
    for pool_name, feature_list in pools_to_evaluate.items():
        print(f"--- Оценка набора признаков: '{pool_name}' ({len(feature_list)} признаков) ---")

        # Отбираем нужные признаки из полных данных
        X_train = X_train_full[feature_list]
        X_val = X_val_full[feature_list]
        
        model_names = [type(m).__name__ for m in models]
        results_df = pd.DataFrame(index=['Train', 'Validation'], columns=model_names)

        # Внутренний цикл по моделям
        for model in models:
            model_name = type(model).__name__
            
            try:
                model.fit(X_train, y_train)
                train_pred_proba = model.predict_proba(X_train)[:, 1]
                val_pred_proba = model.predict_proba(X_val)[:, 1]
                train_score = scoring_func(y_train, train_pred_proba)
                val_score = scoring_func(y_val, val_pred_proba)
            except Exception as e:
                print(f"    ! Ошибка при оценке модели {model_name}: {e}")
                train_score, val_score = 'Error', 'Error'

            results_df.loc['Train', model_name] = train_score
            results_df.loc['Validation', model_name] = val_score

        # Выводим и сохраняем результат для текущего набора признаков
        print(f"\nРезультаты для '{pool_name}' (метрика: {scoring_func.__name__}):")
        display(results_df.round(3))

def custom_gini(y_true, y_prob) -> float:
    """
    Кастомная реализация расчета ROC AUC.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # --- 1. Сортировка объектов по убыванию вероятностей ---
    desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    y_prob = y_prob[desc_score_indices]

    # --- 2. Расчет TPR и FPR в точках изменения порога ---
    # Находим уникальные значения вероятностей, чтобы определить пороги
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Накапливаем количество True Positives и False Positives при движении порога
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Нормализуем, чтобы получить TPR и FPR
    # Добавляем точку (0, 0) в начало
    tpr = np.r_[0, tps] / (np.sum(y_true) if np.sum(y_true) > 0 else 1)
    fpr = np.r_[0, fps] / ((len(y_true) - np.sum(y_true)) if (len(y_true) - np.sum(y_true)) > 0 else 1)

    # --- 3. Расчет площади под кривой (метод трапеций) ---
    # np.trapz(y, x) вычисляет интеграл по точкам (x, y)
    roc_auc = np.trapz(tpr, fpr)
    
    return 2 * roc_auc - 1

def remove_linear_dependencies(df):
    """
    Удаляет любые линейно зависимые столбцы из датафрейма, используя ранг матрицы.

    Args:
        df (pd.DataFrame): Исходный датафрейм с числовыми признаками.

    Returns:
        pd.DataFrame: Датафрейм без линейно зависимых столбцов.
    """
    # Работаем только с числовыми столбцами
    df_numeric = df.select_dtypes(include=np.number)
    
    # 1. Вычисляем ранг исходной матрицы
    initial_rank = matrix_rank(df_numeric.values)
    
    cols_to_keep = list(df_numeric.columns)
    dropped_cols = []

    # 2. Итеративно проверяем каждый столбец
    for col in df_numeric.columns:
        # Временный набор столбцов без текущего
        temp_cols = [c for c in cols_to_keep if c != col]
        
        # 3. Если ранг не изменился после удаления столбца, значит он был "лишним"
        if matrix_rank(df_numeric[temp_cols].values) == initial_rank:
            cols_to_keep.remove(col)
            dropped_cols.append(col)
    
    if dropped_cols:
        print(f"Обнаружены и удалены следующие линейно зависимые столбцы: {dropped_cols}")
    else:
        print("Линейно зависимых столбцов не найдено.")
    
    # Возвращаем исходный датафрейм с нужными столбцами
    return df[cols_to_keep]

def remove_collinear_features(df, threshold=0.9):
    """
    Находит и удаляет мультиколлинеарные признаки из датафрейма.

    Для каждой группы признаков, где попарная корреляция превышает порог,
    оставляет только один признак, а остальные удаляет.

    Args:
        df (pd.DataFrame): Исходный датафрейм с числовыми признаками.
        threshold (float): Порог корреляции (по модулю) для удаления. 
                           Значение 0.999 подходит для борьбы с идеальной мультиколлинеарностью.

    Returns:
        pd.DataFrame: Датафрейм с удаленными признаками.
    """
    # Убедимся, что работаем только с числовыми данными
    df_numeric = df.select_dtypes(include=np.number)
    
    # 1. Рассчитываем матрицу корреляций
    corr_matrix = df_numeric.corr().abs()

    # 2. Берем верхний треугольник матрицы
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 3. Находим столбцы для удаления
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"Удалены следующие столбцы из пары с корреляцией выше {threshold}: {to_drop}")
    else:
        print("Мультиколлинеарных столбцов с порогом >", threshold, "не найдено.")

    # 4. Удаляем столбцы и возвращаем результат
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced

def logistic_regression_feature_selection(train_df, val_df, target_col, p_threshold=0.05, threshold = 0.9, l1_C=1.0):
    """
    Сравнивает три подхода к логистической регрессии: базовый, с отбором по p-value и с L1 регуляризацией.

    Args:
        train_df (pd.DataFrame): Обучающий датафрейм.
        val_df (pd.DataFrame): Валидационный датафрейм.
        target_col (str): Название целевой колонки.
        p_threshold (float): Порог p-value для отбора признаков.
        l1_C (float): Параметр C для L1 регуляризации.

    Returns:
        pd.DataFrame: Итоговая таблица с результатами.
    """
    # Собираем результаты в список словарей
    results_list = []

    # 1. Подготовка данных
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # --- Модель 1: Baseline (все признаки) ---
    print("Обучение Baseline модели...")
    model_full = LogisticRegression(penalty = None, random_state=42)
    model_full.fit(X_train, y_train)
    results_list.append({
        'Model': 'Baseline (All Features)',
        'Train Gini': gini_score_sklearn(y_train, model_full.predict_proba(X_train)[:, 1]),
        'Validation Gini': gini_score_sklearn(y_val, model_full.predict_proba(X_val)[:, 1]),
        'Num Features': X_train.shape[1]
    })

    # --- Модель 2: Отбор по p-value ---
    print(f"Отбор признаков по p-value < {p_threshold}...")
    # Используем statsmodels для получения p-values
    X_train_sm = sm.add_constant(X_train) # statsmodels требует константу (intercept)
    X_train_sm = remove_collinear_features(remove_linear_dependencies(X_train_sm), threshold = threshold)
    logit_model = sm.Logit(y_train, X_train_sm).fit(disp=0, maxiter=1000, method='cg') # disp=0 отключает вывод
    p_values = logit_model.pvalues.drop('const')
    
    selected_features = p_values[p_values < p_threshold].index.tolist()
    
    if not selected_features:
        print(f"Предупреждение: ни один признак не прошел порог p-value {p_threshold}. Используются все признаки.")
        selected_features = X_train.columns.tolist()

    model_pval = LogisticRegression(penalty = None, random_state=42)
    model_pval.fit(X_train[selected_features], y_train)
    results_list.append({
        'Model': 'P-value Selection',
        'Train Gini': gini_score_sklearn(y_train, model_pval.predict_proba(X_train[selected_features])[:, 1]),
        'Validation Gini': gini_score_sklearn(y_val, model_pval.predict_proba(X_val[selected_features])[:, 1]),
        'Num Features': len(selected_features)
    })

    # --- Модель 3: L1 регуляризация (Lasso) ---
    print(f"Обучение модели с L1 регуляризацией (C={l1_C})...")
    model_l1 = LogisticRegression(penalty='l1', C=l1_C, random_state=42, solver = "liblinear")
    model_l1.fit(X_train, y_train)
    
    # Считаем количество ненулевых коэффициентов
    num_l1_features = np.sum(model_l1.coef_ != 0)
    selected_features_by_l1 = X_train.columns[(model_l1.coef_ != 0).flatten()].tolist()
    
    results_list.append({
        'Model': 'L1 Regularization',
        'Train Gini': gini_score_sklearn(y_train, model_l1.predict_proba(X_train)[:, 1]),
        'Validation Gini': gini_score_sklearn(y_val, model_l1.predict_proba(X_val)[:, 1]),
        'Num Features': num_l1_features
    })

    # --- Формирование итоговой таблицы ---
    results_df = pd.DataFrame(results_list).set_index('Model')

    display(results_df)

    return selected_features, selected_features_by_l1


# --- Шаг 2: Основная функция ---

def grid_search_cv_and_validate(model, param_grid, train_df, val_df, target_column):
    """
    Выполняет GridSearchCV с кросс-валидацией на train, отбирает топ-5 моделей,
    и находит лучшую из них по Gini на отложенной validation выборке.

    Args:
        model: Экземпляр модели sklearn.
        param_grid (dict): Словарь с сеткой параметров для перебора.
        train_df (pd.DataFrame): Обучающий датафрейм.
        val_df (pd.DataFrame): Валидационный датафрейм.
        target_column (str): Название целевой колонки.

    Returns:
        dict: Словарь с лучшими найденными параметрами.
    """
    gini_scorer = make_scorer(gini_score_sklearn, greater_is_better=True, response_method='predict_proba')
    # 1. Подготовка данных
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    # 2. Запуск GridSearchCV на обучающей выборке
    grid = GridSearchCV(model, param_grid, scoring=gini_scorer, cv=5, refit=False, return_train_score=True)
    
    print(f"Запускаю GridSearchCV для '{type(model).__name__}'...")
    grid.fit(X_train, y_train)

    # 3. Отбор топ-5 моделей по среднему Gini на кросс-валидации
    results = pd.DataFrame(grid.cv_results_)
    top5_results = results.sort_values(by='mean_test_score', ascending=False).head(5)

    best_val_score = -1
    best_params = {}

    print("\n--- Проверка топ-5 моделей на валидационном сете ---")
    
    # 4. Проверка топ-5 на валидационном сете
    for index, row in top5_results.iterrows():
        params = row['params']
        
        # Обучаем модель с текущими параметрами на train сете
        current_model = model.set_params(**params)
        current_model.fit(X_train, y_train)
        
        # Оцениваем на validation сете
        val_pred_proba = current_model.predict_proba(X_val)[:, 1]
        current_val_score = gini_score_sklearn(y_val, val_pred_proba)
        
        print(f"Параметры: {params}")
        print(f"  - Средний Gini на CV (train): {row['mean_test_score']:.4f}")
        print(f"  - Gini на Validation:          {current_val_score:.4f}")
        
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_params = params

    # 5. Обучаем и выводим результаты для финальной лучшей модели
    print("\n--- Финальная лучшая модель (по Gini на Validation) ---")
    final_model = model.set_params(**best_params)
    final_model.fit(X_train, y_train)
    
    train_final_gini = gini_score_sklearn(y_train, final_model.predict_proba(X_train)[:, 1])
    
    print(f"Лучшие параметры: {best_params}")
    print(f"Gini на Train:      {train_final_gini:.4f}")
    print(f"Gini на Validation: {best_val_score:.4f}")

    return best_params


def evaluate_final_model(model, train_df, val_df, test_df, target_column):
    """
    Обучает финальную модель на полном train сете и оценивает ее 
    производительность на train, validation и test сетах.

    Args:
        model: Экземпляр модели sklearn с уже установленными лучшими гиперпараметрами.
        train_df (pd.DataFrame): Обучающий датафрейм.
        val_df (pd.DataFrame): Валидационный датафрейм.
        test_df (pd.DataFrame): Тестовый датафрейм.
        target_column (str): Название целевой колонки.

    """
    # 1. Подготовка данных для каждого набора
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    print(f"Обучение финальной модели '{type(model).__name__}'...")
    # 2. Обучение модели на всем обучающем наборе
    model.fit(X_train, y_train)

    # 3. Получение вероятностей для всех трех наборов
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    # 4. Расчет Gini для каждого набора
    gini_train = gini_score_sklearn(y_train, train_pred_proba)
    gini_val = gini_score_sklearn(y_val, val_pred_proba)
    gini_test = gini_score_sklearn(y_test, test_pred_proba)

    # 5. Формирование и возврат результата в виде таблицы
    results = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Gini': [gini_train, gini_val, gini_test]
    }).set_index('Dataset')

    print("\n--- Финальная оценка Gini ---")
    display(results.round(3))

def evaluate_classification_metrics(train_df, val_df, test_df, target_column, models, feature_pools=None):
    """
    Обучает и оценивает модели по метрикам Recall, Precision, F1 и AUC PR на train, val и test сетах.

    Args:
        train_df (pd.DataFrame): Обучающий датафрейм.
        val_df (pd.DataFrame): Валидационный датафрейм.
        test_df (pd.DataFrame): Тестовый датафрейм.
        target_column (str): Название целевой колонки.
        models (list): Список экземпляров моделей для оценки.
        feature_pools (dict, optional): Словарь вида {название_пула: список_признаков}.

    Returns:
        dict: Словарь, где ключ - название пула, а значение - DataFrame с результатами.
    """
    # 1. Подготовка полных данных
    X_train_full = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val_full = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    X_test_full = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # 2. Формирование наборов признаков для оценки
    pools_to_evaluate = {'Full Features': X_train_full.columns.tolist()}
    if feature_pools:
        pools_to_evaluate.update(feature_pools)

    # 3. Цикл по наборам признаков
    for pool_name, feature_list in pools_to_evaluate.items():
        print(f"--- Оценка набора признаков: '{pool_name}' ({len(feature_list)} признаков) ---")

        X_train, X_val, X_test = X_train_full[feature_list], X_val_full[feature_list], X_test_full[feature_list]
        
        # Создаем пустой DataFrame с мульти-индексом для результатов
        model_names = [type(m).__name__ for m in models]
        metrics = ['Recall', 'Precision', 'F1-score', 'AUC PR']
        datasets = ['Train', 'Validation', 'Test']
        
        results_df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([model_names, datasets], names=['Model', 'Dataset']),
            index=metrics
        )

        # 4. Цикл по моделям
        for model in models:
            model_name = type(model).__name__
            
            try:
                # Обучение
                model.fit(X_train, y_train)

                # Цикл по датасетам для оценки
                for X, y, dataset_name in [(X_train, y_train, 'Train'), 
                                           (X_val, y_val, 'Validation'), 
                                           (X_test, y_test, 'Test')]:
                    
                    # Предсказания классов (для Recall, Precision, F1)
                    y_pred = model.predict(X)
                    # Предсказания вероятностей (для AUC PR)
                    y_pred_proba = model.predict_proba(X)[:, 1]

                    # Расчет метрик
                    results_df.loc['Recall', (model_name, dataset_name)] = recall_score(y, y_pred)
                    results_df.loc['Precision', (model_name, dataset_name)] = precision_score(y, y_pred)
                    results_df.loc['F1-score', (model_name, dataset_name)] = f1_score(y, y_pred)
                    results_df.loc['AUC PR', (model_name, dataset_name)] = average_precision_score(y, y_pred_proba)

            except Exception as e:
                print(f"    ! Ошибка при оценке модели {model_name}: {e}")
   
        display(results_df.round(4))
            