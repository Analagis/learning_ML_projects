import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_true_vs_pred_subplots(
    y_val_true: pd.DataFrame,
    y_val_pred: pd.DataFrame,
    areas: list,
    n_cols: int = 2,
    figsize=(16, 8)
):
    """
    y_val_true, y_val_pred: DataFrame, одинаковый индекс (время) и колонки = участки
    areas: список колонок (участков), которые нужно нарисовать
    n_cols: сколько графиков в строке
    """

    n = len(areas)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)  # на случай 1 строки/1 графика

    for i, area in enumerate(areas):
        ax = axes[i]
        ax.plot(y_val_true.index, y_val_true[area], label="true", color="black", linewidth=1)
        ax.plot(y_val_pred.index, y_val_pred[area], label="pred", color="tab:blue", linewidth=1)
        ax.set_title(area, fontsize=10)
        ax.grid(True, alpha=0.3)

    # скрываем пустые оси, если areas меньше, чем ячеек
    for j in range(len(areas), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    plt.show()

def build_time_features(
    df,
    timestamp_col="Trip Start Timestamp",
    daily_period=96,
    fourier_order=2
):
    """
    Строит календарные и Фурье-признаки для временного ряда с суточной сезонностью.

    Аргументы:
        df : pd.DataFrame
            Исходный датафрейм, содержащий колонку с датой/временем и колонки районов.
        timestamp_col : str
            Имя колонки с меткой времени.
        daily_period : int
            Длина суточного периода в шагах (для 15-минутных данных: 24*4 = 96).
        fourier_order : int
            Порядок Фурье-рядов (число гармоник k, для которых строим sin/cos).

    Возвращает:
        features_df : pd.DataFrame
    """
    features_df = df[[timestamp_col]].copy()
    features_df[timestamp_col] = pd.to_datetime(features_df[timestamp_col])

    dt = features_df[timestamp_col].dt

    # Календарные признаки
    features_df["day_of_week"] = dt.dayofweek  # 0=Mon, 6=Sun
    features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6]).astype(int)

    features_df["hour"] = dt.hour

    bins = [-1, 7, 10, 16, 21, 23]
    labels = ["night", "morning_peak", "day", "evening_peak", "late_evening"]
    features_df = pd.concat(
        [features_df.drop(columns=["hour"]), pd.get_dummies(
        pd.cut(
        features_df["hour"],
        bins=bins,
        labels=labels
    ),
        prefix="hbin",
        drop_first=True
    )*1],
        axis=1
    )

    features_df = pd.concat(
    [features_df.drop(columns=["day_of_week"]), pd.get_dummies(
    features_df["day_of_week"],
    prefix="dow",
    drop_first=True  # чтобы не было идеальной коллинеарности
)*1],
    axis=1
)

    # Индекс времени для Фурье (0,1,2,...)
    t = np.arange(len(features_df))

    # Ряды Фурье для суточной сезонности
    for k in range(1, fourier_order + 1):
        angle = 2 * np.pi * k * t / daily_period
        features_df[f"fourier_sin_{k}"] = np.sin(angle)
        features_df[f"fourier_cos_{k}"] = np.cos(angle)

    return features_df

def build_ar_ma_features(
    df: pd.DataFrame,
    timestamp_col: str = "Trip Start Timestamp",
    lag_list: list[int] = None,
    window_list: list[int] = None,
    agg_funcs: dict = None,
) -> pd.DataFrame:
    """
    Строит AR (лаговые) и MA-подобные (rolling-агрегаты) признаки
    для всех районов в широком датафрейме и возвращает long-формат.

    Параметры
    ---------
    df : pd.DataFrame
        Широкий датафрейм:
        [timestamp_col, area_0, area_1, ..., area_N].
    timestamp_col : str
        Имя колонки с меткой времени.
    lag_list : List[int]
        Список лагов для AR-признаков, например [1, 2, 3, 96].
    window_list : List[int]
        Список размеров окон для rolling-признаков, например [4, 96].
    agg_funcs : Dict[str, Callable]
        Словарь агрегаторов для rolling:
        ключ = суффикс в названии признака (например, "median", "std"),
        значение = функция (например, np.median, np.std).

    Возвращает
    ----------
    features_long : pd.DataFrame
        Long-формат:
        - timestamp (как timestamp_col),
        - area (имя района),
        - target (исходный ряд),
        - y_lag_{k},
        - roll_{agg_name}_{window}
    """

    if lag_list is None:
        lag_list = []
    if window_list is None:
        window_list = []
    if agg_funcs is None:
        agg_funcs = {}

    # Копия, чтобы не портить исходный df
    data = df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    # Список колонок районов (всё, кроме timestamp_col)
    area_cols = [c for c in data.columns if c != timestamp_col]

    # Здесь будем накапливать расширенный "широкий" df по районам
    wide_with_features = []

    for area in area_cols:
        tmp = data[[timestamp_col, area]].copy()
        tmp = tmp.rename(columns={area: "target"})
        # Лаги (AR-признаки)
        for lag in lag_list:
            col_name = f"y_lag_{lag}"
            tmp[col_name] = tmp["target"].shift(lag)

        # Rolling-агрегаты (MA-подобные признаки)
        for window in window_list:
            for agg_name, agg_func in agg_funcs.items():
                col_name = f"roll_{agg_name}_{window}"
                tmp[col_name] = (
                    tmp["target"]
                    .rolling(window=window, min_periods=1)
                    .apply(agg_func, raw=True)
                )

        # Добавляем идентификатор района
        tmp["area"] = area

        wide_with_features.append(tmp)

    # Собираем все районы обратно
    features_long = pd.concat(wide_with_features, axis=0, ignore_index=True)

    # Можно отсечь первые строки, где нет всех лагов
    max_lag = max(lag_list) if lag_list else 0
    max_window = max(window_list) if window_list else 0
    drop_n = max(max_lag, max_window)
    if drop_n > 0:
        features_long["row_in_group"] = features_long.groupby("area").cumcount()
        features_long = features_long[features_long["row_in_group"] >= drop_n]
        features_long = features_long.drop(columns=["row_in_group"]).reset_index(drop=True)

    return features_long