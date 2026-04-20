"""
Deep-learning replacement for MLModel.

Expected to be provided in the module namespace by the caller (same pattern
as model.py): `get_user_defined_score`, `ReverseSmooth`, `Postprocessing`,
`month_format`. Either import them here, or inject via
`import model_dl; model_dl.get_user_defined_score = ...` before calling
DLModel.
"""
import math
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


GI_LAG_PATTERN = re.compile(r'^GI \(t-(\d+)\)$')
MOU_LAG_PATTERN = re.compile(r'^MOU_t-(\d+)$')


def identify_feature_groups(columns):
    """Split X columns into (gi_lags_chronological, mou_lags_chronological, scalars)."""
    gi_pairs, mou_pairs, scalars = [], [], []
    for col in columns:
        m = GI_LAG_PATTERN.match(col)
        if m:
            gi_pairs.append((int(m.group(1)), col))
            continue
        m = MOU_LAG_PATTERN.match(col)
        if m:
            mou_pairs.append((int(m.group(1)), col))
            continue
        scalars.append(col)
    # oldest first: t-12 -> t-1
    gi_cols = [c for _, c in sorted(gi_pairs, key=lambda x: -x[0])]
    mou_cols = [c for _, c in sorted(mou_pairs, key=lambda x: -x[0])]
    return gi_cols, mou_cols, scalars


class PartsDataset(Dataset):
    def __init__(self, X, y, gi_cols, mou_cols, scalar_cols):
        self.gi = X[gi_cols].to_numpy(dtype=np.float32)
        self.mou = X[mou_cols].to_numpy(dtype=np.float32)
        if scalar_cols:
            self.scal = X[scalar_cols].to_numpy(dtype=np.float32)
        else:
            self.scal = np.zeros((len(X), 0), dtype=np.float32)
        self.y = y.to_numpy(dtype=np.float32) if y is not None else None

    def __len__(self):
        return len(self.gi)

    def __getitem__(self, idx):
        gi = torch.from_numpy(self.gi[idx]).unsqueeze(-1)
        mou = torch.from_numpy(self.mou[idx]).unsqueeze(-1)
        scal = torch.from_numpy(self.scal[idx])
        if self.y is None:
            return gi, mou, scal
        return gi, mou, scal, torch.from_numpy(self.y[idx])


class DualEncoderLSTM(nn.Module):
    def __init__(self, n_scalars, hidden=64, dropout=0.2, n_outputs=6):
        super().__init__()
        self.gi_lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.mou_lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        fusion_in = 2 * hidden + n_scalars
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, gi_seq, mou_seq, scalars):
        _, (h_gi, _) = self.gi_lstm(gi_seq)
        _, (h_mou, _) = self.mou_lstm(mou_seq)
        feat = torch.cat([h_gi[-1], h_mou[-1], scalars], dim=-1)
        return self.head(feat)


def make_recency_weights(X_train, min_w=1.0, max_w=20.0):
    """Linear weights over distinct versions: oldest -> min_w, most recent -> max_w."""
    versions = X_train.index.get_level_values('version')
    version_sorted = sorted(pd.unique(versions))
    n = len(version_sorted)
    if n == 1:
        rank_weight = {version_sorted[0]: max_w}
    else:
        rank_weight = {v: min_w + (max_w - min_w) * i / (n - 1) for i, v in enumerate(version_sorted)}
    return np.array([rank_weight[v] for v in versions], dtype=np.float64)


@torch.no_grad()
def predict_all(model, loader, has_y):
    model.eval()
    preds = []
    for batch in loader:
        if has_y:
            gi, mou, scal, _ = batch
        else:
            gi, mou, scal = batch
        gi = gi.to(device)
        mou = mou.to(device)
        scal = scal.to(device)
        preds.append(model(gi, mou, scal).cpu().numpy())
    return np.concatenate(preds, axis=0)


def DLModel(
    df_gi: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, test_date: str, validation_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str]], float, float]:
    """
    Deep-learning drop-in for MLModel.

    Architecture: dual-encoder LSTM (GI seq + MOU seq) concatenated with scalar
    features through a 3-layer MLP head producing 6 outputs. Train/val split
    is the LAST fold of TimeSeriesSplit(3); recency-weighted sampling with
    linear weights 1 -> 20 (oldest -> most recent version). No HP tuning.
    Post-processing block is copied verbatim from MLModel.
    """

    # --- 1. Feature groups ---
    gi_cols, mou_cols, scalar_cols = identify_feature_groups(list(X_train.columns))
    n_scalars = len(scalar_cols)

    # --- 2. Sort by version (same as MLModel) ---
    X_train = X_train.sort_index(level='version')
    y_train = y_train.sort_index(level='version')

    # --- 3. Last fold of TimeSeriesSplit ---
    tscv = TimeSeriesSplit(n_splits=3)
    last_train_idx, last_val_idx = list(tscv.split(X_train))[-1]
    X_tr, y_tr = X_train.iloc[last_train_idx], y_train.iloc[last_train_idx]
    X_val, y_val = X_train.iloc[last_val_idx], y_train.iloc[last_val_idx]

    # --- 4. Datasets and loaders ---
    hparams = {
        'architecture': 'DualEncoderLSTM',
        'hidden': 64,
        'dropout': 0.2,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 50,
        'patience': 8,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'recency_min_weight': 1.0,
        'recency_max_weight': 20.0,
    }

    train_ds = PartsDataset(X_tr, y_tr, gi_cols, mou_cols, scalar_cols)
    val_ds = PartsDataset(X_val, y_val, gi_cols, mou_cols, scalar_cols)
    test_ds = PartsDataset(X_test, None, gi_cols, mou_cols, scalar_cols)

    weights = make_recency_weights(X_tr, hparams['recency_min_weight'], hparams['recency_max_weight'])
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], sampler=sampler)
    train_loader_eval = DataLoader(train_ds, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # --- 5. Model, optimizer, loss ---
    model = DualEncoderLSTM(n_scalars=n_scalars, hidden=hparams['hidden'], dropout=hparams['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    loss_fn = nn.MSELoss()

    # --- 6. Training loop with early stopping on validation score ---
    print('-' * 15, 'DL Training', '-' * 15)
    best_val = -math.inf
    bad_epochs = 0
    best_state = None
    for epoch in range(hparams['epochs']):
        model.train()
        for gi, mou, scal, y in train_loader:
            gi = gi.to(device)
            mou = mou.to(device)
            scal = scal.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(gi, mou, scal), y)
            loss.backward()
            optimizer.step()

        val_preds = predict_all(model, val_loader, has_y=True)
        val_score = get_user_defined_score(y_val.values, val_preds)
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= hparams['patience']:
                print(f'Early stopping at epoch {epoch + 1} (best val score = {best_val:.4f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 7. Final train and validation scores ---
    train_preds = predict_all(model, train_loader_eval, has_y=True)
    cv_train_score = float(get_user_defined_score(y_tr.values, train_preds))
    val_preds = predict_all(model, val_loader, has_y=True)
    cv_validation_score = float(get_user_defined_score(y_val.values, val_preds))

    # --- 8. Prediction on X_test ---
    test_preds = predict_all(model, test_loader, has_y=False)

    start_date_ts = pd.to_datetime(test_date, format=month_format)
    prediction_dates = pd.date_range(
        start=start_date_ts,
        periods=len(validation_columns),
        freq='MS',
    ).strftime(month_format)

    pred_df = pd.DataFrame(test_preds, index=X_test.index, columns=prediction_dates)
    if not isinstance(pred_df.index, pd.MultiIndex) or pred_df.index.names != ['part', 'plant', 'version']:
        pred_df.index = pd.MultiIndex.from_tuples(pred_df.index, names=['part', 'plant', 'version'])

    # --- 9. Post-processing (copied verbatim from MLModel) ---
    gi_data = df_gi.reorder_levels(['plant', 'part'])
    gi_data.columns.name = 'version'
    gi_data.columns = pd.to_datetime(gi_data.columns).strftime(month_format)
    pred_df = pred_df.reorder_levels(['version', 'plant', 'part'])
    pred_df.index = pred_df.index.set_levels([pred_df.index.levels[0].to_series().dt.strftime(month_format), pred_df.index.levels[1].astype(str), pred_df.index.levels[2].astype(str)])

    pred_df_reverse = pd.concat(list(pred_df.apply(lambda x: ReverseSmooth(gi_data=gi_data, pred=x), axis=1)), axis=1).T
    print('Smoothing Reversed\n- End -')
    pred_df_reverse.index = pd.MultiIndex.from_tuples(pred_df_reverse.index.str.split('|').tolist()).set_names(['version', 'plant', 'part'])
    pred_df_final = Postprocessing(pred_df_reverse)
    pred_df_final.columns = validation_columns

    return pred_df_final, hparams, cv_train_score, cv_validation_score
