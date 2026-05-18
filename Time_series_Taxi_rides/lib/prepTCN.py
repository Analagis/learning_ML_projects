import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from tsai.all import *
from torch.utils.data import Dataset, DataLoader

import json
import pickle
from pathlib import Path

def split_wide_by_time(df, timestamp_col="Trip Start Timestamp", horizon=673):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    test = df.iloc[n - horizon:].copy()
    val = df.iloc[n - 2 * horizon:n - horizon].copy()
    train = df.iloc[:n - 2 * horizon].copy()

    return train, val, test


def fit_area_scalers(train_df, area_cols):
    scalers = {}
    for col in area_cols:
        series = train_df[col].astype(float).values
        mean_ = series.mean()
        std_ = series.std()
        if std_ == 0:
            std_ = 1.0
        scalers[col] = {"mean": mean_, "std": std_}
    return scalers


def transform_wide_df(df, area_cols, scalers):
    df_scaled = df.copy()
    for col in area_cols:
        mean_ = scalers[col]["mean"]
        std_ = scalers[col]["std"]
        df_scaled[col] = (df_scaled[col].astype(float) - mean_) / std_
    return df_scaled


def inverse_transform_series(values, area_name, scalers):
    mean_ = scalers[area_name]["mean"]
    std_ = scalers[area_name]["std"]
    return values * std_ + mean_


def build_tcn_train_windows(train_scaled, area_cols, context_len=192):
    X_list = []
    y_list = []

    for col in area_cols:
        series = train_scaled[col].astype(np.float32).values

        for t in range(context_len, len(series)):
            x = series[t - context_len:t]
            y = series[t]

            X_list.append(x[None, :])   # shape: (1, context_len)
            y_list.append(y)

    X = np.stack(X_list).astype(np.float32)   # (n_samples, 1, context_len)
    y = np.array(y_list, dtype=np.float32)    # (n_samples,)

    return X, y


def build_tcn_val_windows(train_scaled, val_scaled, area_cols, context_len=192):
    X_list = []
    y_list = []

    for col in area_cols:
        train_series = train_scaled[col].astype(np.float32).values
        val_series = val_scaled[col].astype(np.float32).values

        history = np.concatenate([train_series, val_series], axis=0)

        for i in range(len(val_series)):
            t = len(train_series) + i
            x = history[t - context_len:t]
            y = history[t]

            X_list.append(x[None, :])
            y_list.append(y)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


class SimpleArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def fit_tcn_tsai(
    X_train,
    y_train,
    X_val,
    y_val,
    context_len,
    bs=256,
    epochs=20,
    lr=1e-3,
    seed=42
):
    set_seed(seed, reproducible=True)

    train_ds = SimpleArrayDataset(X_train, y_train)
    valid_ds = SimpleArrayDataset(X_val, y_val)

    dls = TSDataLoaders.from_dsets(
        train_ds,
        valid_ds,
        bs=bs,
        batch_tfms=None,
        shuffle_train=True
    )

    model = TCN(
        c_in=1,
        c_out=1,
        layers=[64, 64, 64],
        ks=3,
        conv_dropout=0.1
    )

    learn = Learner(
        dls,
        model,
        loss_func=MSELossFlat(),
        metrics=[mae]
    )

    learn.fit_one_cycle(epochs, lr)

    return learn


def recursive_forecast_area(model, history_scaled, horizon, context_len):
    preds = []
    history = history_scaled.astype(np.float32).copy()

    for _ in range(horizon):
        x = history[-context_len:]
        x = x[None, None, :]  # (1, 1, context_len)

        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=next(model.parameters()).device)
            pred = model(x_t).detach().cpu().numpy().reshape(-1)[0]

        preds.append(pred)
        history = np.append(history, pred)

    return np.array(preds, dtype=np.float32)


def forecast_wide_recursive(
    learn,
    train_scaled,
    val_scaled,
    test_scaled,
    area_cols,
    scalers,
    context_len=192,
    horizon=673
):
    val_pred_scaled = {}
    test_pred_scaled = {}

    model = learn.model.eval()

    for col in area_cols:
        train_series = train_scaled[col].astype(np.float32).values
        val_series = val_scaled[col].astype(np.float32).values
        test_series = test_scaled[col].astype(np.float32).values

        val_pred_scaled[col] = recursive_forecast_area(
            model=model,
            history_scaled=train_series,
            horizon=horizon,
            context_len=context_len
        )

        history_for_test = np.concatenate([train_series, val_series], axis=0)
        test_pred_scaled[col] = recursive_forecast_area(
            model=model,
            history_scaled=history_for_test,
            horizon=horizon,
            context_len=context_len
        )

    val_pred_df = pd.DataFrame(val_pred_scaled, index=val_scaled.index)
    test_pred_df = pd.DataFrame(test_pred_scaled, index=test_scaled.index)

    for col in area_cols:
        val_pred_df[col] = inverse_transform_series(val_pred_df[col].values, col, scalers)
        test_pred_df[col] = inverse_transform_series(test_pred_df[col].values, col, scalers)

    return val_pred_df, test_pred_df


def evaluate_mae_by_area(true_df, pred_df, area_cols):
    rows = []
    for col in area_cols:
        mae_val = mean_absolute_error(true_df[col].values, pred_df[col].values)
        rows.append({"area": col, "MAE_val": mae_val})
    return pd.DataFrame(rows).sort_values("MAE_val", ascending=False).reset_index(drop=True)


def prepare_and_train_tcn_pipeline(
    df,
    timestamp_col="Trip Start Timestamp",
    horizon=673,
    context_len=192,
    bs=256,
    epochs=20,
    lr=1e-3,
    seed=42
):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    area_cols = [c for c in df.columns if c != timestamp_col]

    train, val, test = split_wide_by_time(df, timestamp_col=timestamp_col, horizon=horizon)

    scalers = fit_area_scalers(train, area_cols)

    train_scaled = transform_wide_df(train, area_cols, scalers)
    val_scaled = transform_wide_df(val, area_cols, scalers)
    test_scaled = transform_wide_df(test, area_cols, scalers)

    X_train, y_train = build_tcn_train_windows(train_scaled, area_cols, context_len=context_len)
    X_val, y_val = build_tcn_val_windows(train_scaled, val_scaled, area_cols, context_len=context_len)

    learn = fit_tcn_tsai(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        context_len=context_len,
        bs=bs,
        epochs=epochs,
        lr=lr,
        seed=seed
    )

    val_pred_df, test_pred_df = forecast_wide_recursive(
        learn=learn,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        area_cols=area_cols,
        scalers=scalers,
        context_len=context_len,
        horizon=horizon
    )

    val_true_df = val[area_cols].reset_index(drop=True)
    test_true_df = test[area_cols].reset_index(drop=True)

    val_pred_df = val_pred_df.reset_index(drop=True)
    test_pred_df = test_pred_df.reset_index(drop=True)

    tcn_info_df = evaluate_mae_by_area(val_true_df, val_pred_df, area_cols)

    return {
        "learn": learn,
        "scalers": scalers,
        "train": train,
        "val": val,
        "test": test,
        "val_true_df": val_true_df,
        "test_true_df": test_true_df,
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
        "tcn_info_df": tcn_info_df,
        "area_cols": area_cols,
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape
    }


def save_tcn_artifacts(result: dict, base_path: str = "tcn_artifacts"):
    """
    Сохраняет артефакты TCN-пайплайна из словаря result.

    Структура:
      base_path/
        tcn_model.pth         - state_dict модели (PyTorch)
        scalers.json          - словарь нормализаций по районам
        train.parquet
        val.parquet
        test.parquet
        val_true_df.parquet
        test_true_df.parquet
        val_pred_df.parquet
        test_pred_df.parquet
        tcn_info_df.parquet
        meta.json             - area_cols, X_train_shape, X_val_shape, model_params
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    learn = result["learn"]
    scalers = result["scalers"]
    train = result["train"]
    val = result["val"]
    test = result["test"]
    val_true_df = result["val_true_df"]
    test_true_df = result["test_true_df"]
    val_pred_df = result["val_pred_df"]
    test_pred_df = result["test_pred_df"]
    tcn_info_df = result["tcn_info_df"]
    area_cols = result["area_cols"]
    X_train_shape = result["X_train_shape"]
    X_val_shape = result["X_val_shape"]

    # 1) Модель: сохраняем только state_dict
    model = learn.model
    torch.save(model.state_dict(), base / "tcn_model.pth")

    # 2) Скейлеры
    with open(base / "scalers.json", "w") as f:
        json.dump(scalers, f)

    # 3) Таблицы
    train.to_parquet(base / "train.parquet")
    val.to_parquet(base / "val.parquet")
    test.to_parquet(base / "test.parquet")

    val_true_df.to_parquet(base / "val_true_df.parquet")
    test_true_df.to_parquet(base / "test_true_df.parquet")
    val_pred_df.to_parquet(base / "val_pred_df.parquet")
    test_pred_df.to_parquet(base / "test_pred_df.parquet")
    tcn_info_df.to_parquet(base / "tcn_info_df.parquet")

    # 4) Мета-информация: тут важно сохранить параметры модели
    #   (они должны совпадать с тем, как ты создавал TCN в fit_tcn_tsai)
    model_params = {
        "c_in": 1,
        "c_out": 1,
        "layers": [64, 64, 64],
        "ks": 3,
        "conv_dropout": 0.1,
    }

    meta = {
        "area_cols": area_cols,
        "X_train_shape": list(X_train_shape),
        "X_val_shape": list(X_val_shape),
        "model_params": model_params,
    }
    with open(base / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Артефакты TCN сохранены в {base.resolve()}")

def load_tcn_artifacts(base_path: str = "tcn_artifacts", device=None) -> dict:
    """
    Загружает артефакты TCN-пайплайна, сохранённые save_tcn_artifacts.

    Возвращает словарь с теми же ключами, что и result,
    где:
      - learn будет собран заново (с пустыми DataLoaders)
      - model уже имеет загруженные веса.
    """
    base = Path(base_path)

    # 1) Скейлеры и мета-информация
    with open(base / "scalers.json", "r") as f:
        scalers = json.load(f)

    with open(base / "meta.json", "r") as f:
        meta = json.load(f)

    area_cols = meta["area_cols"]
    X_train_shape = tuple(meta["X_train_shape"])
    X_val_shape = tuple(meta["X_val_shape"])
    model_params = meta["model_params"]

    # 2) Модель
    model = TCN(
        c_in=model_params["c_in"],
        c_out=model_params["c_out"],
        layers=model_params["layers"],
        ks=model_params["ks"],
        conv_dropout=model_params["conv_dropout"],
    )

    state_dict = torch.load(base / "tcn_model.pth", map_location=device or "cpu")
    model.load_state_dict(state_dict)
    model.to(device or "cpu")
    model.eval()

    # 3) Таблицы
    train = pd.read_parquet(base / "train.parquet")
    val = pd.read_parquet(base / "val.parquet")
    test = pd.read_parquet(base / "test.parquet")

    val_true_df = pd.read_parquet(base / "val_true_df.parquet")
    test_true_df = pd.read_parquet(base / "test_true_df.parquet")
    val_pred_df = pd.read_parquet(base / "val_pred_df.parquet")
    test_pred_df = pd.read_parquet(base / "test_pred_df.parquet")
    tcn_info_df = pd.read_parquet(base / "tcn_info_df.parquet")

    # 4) (Опционально) собираем "пустой" Learner, если он тебе нужен.
    #    Для inference достаточно иметь model; Learner полезен для дообучения.
    learn = None

    loaded = {
        "learn": learn,           # при желании можно собрать Learner вручную
        "model": model,           # основное, что нужно для прогнозов
        "scalers": scalers,
        "train": train,
        "val": val,
        "test": test,
        "val_true_df": val_true_df,
        "test_true_df": test_true_df,
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
        "tcn_info_df": tcn_info_df,
        "area_cols": area_cols,
        "X_train_shape": X_train_shape,
        "X_val_shape": X_val_shape,
    }

    print(f"Артефакты TCN загружены из {base.resolve()}")
    return loaded