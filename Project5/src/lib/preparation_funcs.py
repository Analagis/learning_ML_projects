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


def apply_feature_scaling(train_df, val_df, test_df, columns=None, fill_na_strategy='median') -> tuple:
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
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

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

    return train_scaled, val_scaled, test_scaled
