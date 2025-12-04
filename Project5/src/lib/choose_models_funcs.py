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

