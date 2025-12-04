import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders import CountEncoder
from sklearn.impute import SimpleImputer

def time_based_split(df: pd.DataFrame, date_col: str) -> tuple:
    """
    Разделяет DataFrame на обучающую, валидационную и тестовую выборки
    в хронологическом порядке по третям.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        date_col (str): Название столбца с датой.

    Returns:
        tuple: Кортеж из трех DataFrame (train, validation, test).
    """
    # 1. Убедимся, что столбец с датой имеет правильный тип и отсортирован
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # 2. Находим уникальные даты и определяем точки разделения
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    
    # Границы для третей
    split_point_1 = unique_dates[n_dates // 3]
    split_point_2 = unique_dates[2 * n_dates // 3]
    
    # 3. Разделяем DataFrame по датам
    train_df = df[df[date_col] < split_point_1]
    validation_df = df[(df[date_col] >= split_point_1) & (df[date_col] < split_point_2)]
    test_df = df[df[date_col] >= split_point_2]
    
    return train_df, validation_df, test_df


def apply_ce_encoding(train_df, val_df, test_df, columns=None) -> tuple:
    """
    Применяет Count Encoding с использованием библиотеки category_encoders.
    """
    # Создаем копии, чтобы не изменять оригинальные датафреймы
    train_encoded = train_df.copy()
    val_encoded = val_df.copy()
    test_encoded = test_df.copy()

    if columns is None:
        columns = train_encoded.select_dtypes(include=['object']).columns.tolist()

    # 1. Создание и обучение кодировщика ТОЛЬКО на train
    encoder = CountEncoder(cols=columns, handle_missing='value')
    encoder.fit(train_encoded)
    
    # 2. Применение ко всем датасетам
    train_encoded = encoder.transform(train_encoded)
    val_encoded = encoder.transform(val_encoded)
    test_encoded = encoder.transform(test_encoded)
    
    # Преобразуем столбцы в int
    for col in columns:
        train_encoded[col] = train_encoded[col].astype(int)
        val_encoded[col] = val_encoded[col].astype(int)
        test_encoded[col] = test_encoded[col].astype(int)

    return train_encoded, val_encoded, test_encoded


def apply_feature_scaling(train_df, val_df, test_df, columns=None, target = "IsBadBuy",fill_na_strategy='median') -> tuple:
    """
    Применяет масштабирование к числовым признакам датафреймов.
    
    Скейлер обучается на обучающем наборе (train_df) и применяется
    к обучающему, валидационному (val_df) и тестовому (test_df) наборам.

    Args:
        train_df (pd.DataFrame): Обучающий датафрейм.
        val_df (pd.DataFrame): Валидационный датафрейм.
        test_df (pd.DataFrame): Тестовый датафрейм.
        columns (list, optional): Список столбцов для масштабирования. 
                                  Если None, масштабируются все числовые столбцы (int, float)

    Returns:
        tuple: Кортеж из трех масштабированных датафреймов (train, validation, test).
    """
    # Создаем копии, чтобы не изменять оригинальные датафреймы
    train_scaled = train_df.drop([target], axis=1).copy()
    val_scaled = val_df.drop([target], axis=1).copy()
    test_scaled = test_df.drop([target], axis=1).copy()

    # Находим столбцы с датой во всех датафреймах
    datetime_cols = train_scaled.select_dtypes(include=['datetime', 'datetimetz']).columns
    if len(datetime_cols) > 0:
        for col in datetime_cols:
            train_scaled[col] = train_scaled[col].astype('int64')
            val_scaled[col] = val_scaled[col].astype('int64')
            test_scaled[col] = test_scaled[col].astype('int64')

    # Если список столбцов не задан, выбираем все числовые столбцы
    if columns is None:
        # Выбираем столбцы с числовыми типами данных
        columns = train_scaled.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    if fill_na_strategy:
        imputer = SimpleImputer(strategy=fill_na_strategy)
        
        # Обучаем imputer ТОЛЬКО на train данных
        imputer.fit(train_scaled[columns])
        
        # Применяем ко всем сетам
        train_scaled[columns] = imputer.transform(train_scaled[columns])
        val_scaled[columns] = imputer.transform(val_scaled[columns])
        test_scaled[columns] = imputer.transform(test_scaled[columns])

    # 1. Обучение: обучаем скейлер ТОЛЬКО на обучающей выборке
    scaler = StandardScaler()
    scaler.fit(train_scaled[columns])

    # 2. Применение: преобразуем все три датасета
    train_scaled[columns] = scaler.transform(train_scaled[columns])
    val_scaled[columns] = scaler.transform(val_scaled[columns])
    test_scaled[columns] = scaler.transform(test_scaled[columns])

    train_scaled[target] = train_df[target]
    val_scaled[target] = val_df[target]
    test_scaled[target] = test_df[target]


    return train_scaled, val_scaled, test_scaled


def generate_nonlinear_features(train_df, val_df, test_df, 
                                frac_rules=None, 
                                groupby_rules=None, 
                                transform_rules=None) -> tuple:
    """
    Создает нелинейные признаки для обучающей, валидационной и тестовой выборок.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Входные датафреймы.
        frac_rules (list of tuples): Правила для создания дробей, e.g., [('num_col', 'denom_col')].
        groupby_rules (dict): Правила для групповых признаков, e.g., 
                              {'new_col_name': ('cat_col', 'cont_col', 'agg_func')}.
        transform_rules (dict): Правила для простых преобразований, e.g., 
                                {'col_name': [np.log1p, np.square]}.

    Returns:
        tuple: Кортеж из трех датафреймов с новыми признаками.
    """
    # Создаем копии, чтобы не изменять оригиналы
    train_new, val_new, test_new = train_df.copy(), val_df.copy(), test_df.copy()
    datasets = [train_new, val_new, test_new]

    # --- 1. Создание дробных признаков ---
    if frac_rules:
        for num_col, denom_col in frac_rules:
            new_col_name = f'{num_col}_div_{denom_col}'
            for df in datasets:
                # Безопасное деление с заменой 0 на NaN, чтобы избежать inf
                df[new_col_name] = df[num_col] / df[denom_col].replace(0, np.nan)

    # --- 2. Создание групповых признаков ---
    if groupby_rules:
        for new_col_name, (cat_col, cont_col, agg_func) in groupby_rules.items():
            # Обучение: считаем агрегаты ТОЛЬКО на train_df
            mapping = train_new.groupby(cat_col)[cont_col].agg(agg_func)
            
            # Применение: отображаем на все датасеты
            for df in datasets:
                df[new_col_name] = df[cat_col].map(mapping)
            
            # Обработка новых категорий: заполняем пропуски глобальной статистикой
            global_stat = train_new[cont_col].agg(agg_func)
            for df in datasets:
                df[new_col_name] = df[new_col_name].fillna(global_stat)

    # --- 3. Создание признаков через простые трансформации ---
    if transform_rules:
        for col, funcs in transform_rules.items():
            for func in funcs:
                new_col_name = f'{col}_{func.__name__}'
                for df in datasets:
                    df[new_col_name] = func(df[col])

    return train_new, val_new, test_new