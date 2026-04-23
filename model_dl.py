"""
Deep-learning replacement for MLModel (v2).

Expected to be provided in the module namespace by the caller (same pattern
as model.py): `get_user_defined_score`, `ReverseSmooth`, `Postprocessing`,
`month_format`. Either import them here, or inject via
`import model_dl; model_dl.get_user_defined_score = ...` before calling
DLModel.

v2 changes vs v1:
- DLModel does its own scaling (caller must pass unscaled X/y; see split.py's
  scale=False path). Rationale: per-column MinMax destroys cross-time
  comparability on GI lags, and v1's asymmetric X-scaled / y-raw setup forced
  the head to learn a 10^4 multiplicative scale.
- Shared log1p + standardize across GI lags AND y targets (single mu/sigma).
- RobustScaler on scalar features.
- MOU lags pass through (already [0,1] upstream).
- 90/10 chronological split by distinct `version` (was: last TimeSeriesSplit fold).
- Optional small HP search (off by default) + final retrain on 100% of training.
"""
import math
import re
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


GI_LAG_PATTERN = re.compile(r'^GI \(t-(\d+)\)$')
MOU_LAG_PATTERN = re.compile(r'^MOU_t-(\d+)$')


DEFAULT_HP_GRID = {
    'hidden': [32, 64],
    'dropout': [0.1, 0.3],
    'lr': [1e-3, 5e-4],
}


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


class DLPreprocessor:
    """
    GI lags + y targets: shared log1p + standardize (single mu/sigma).
    Scalars: RobustScaler.
    MOU lags: pass-through.
    """

    def __init__(self):
        self.gi_cols: List[str] = []
        self.mou_cols: List[str] = []
        self.scalar_cols: List[str] = []
        self.y_cols: List[str] = []
        self.gi_mu: float = 0.0
        self.gi_sigma: float = 1.0
        self.scalar_scaler: Optional[RobustScaler] = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, gi_cols, mou_cols, scalar_cols):
        self.gi_cols = list(gi_cols)
        self.mou_cols = list(mou_cols)
        self.scalar_cols = list(scalar_cols)
        self.y_cols = list(y.columns)

        gi_vals = X[self.gi_cols].to_numpy(dtype=np.float64).ravel()
        y_vals = y.to_numpy(dtype=np.float64).ravel()
        union = np.concatenate([gi_vals, y_vals])
        union = np.log1p(np.clip(union, a_min=0.0, a_max=None))
        self.gi_mu = float(union.mean())
        self.gi_sigma = float(union.std())
        if self.gi_sigma < 1e-8:
            self.gi_sigma = 1.0

        if self.scalar_cols:
            self.scalar_scaler = RobustScaler()
            self.scalar_scaler.fit(X[self.scalar_cols].to_numpy(dtype=np.float64))
        return self

    def _transform_gi_block(self, arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, a_min=0.0, a_max=None)
        return (np.log1p(arr) - self.gi_mu) / self.gi_sigma

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame]):
        X_out = pd.DataFrame(index=X.index)
        X_out[self.gi_cols] = self._transform_gi_block(X[self.gi_cols].to_numpy(dtype=np.float64))
        if self.mou_cols:
            X_out[self.mou_cols] = X[self.mou_cols].astype(np.float64).to_numpy()
        if self.scalar_cols:
            X_out[self.scalar_cols] = self.scalar_scaler.transform(X[self.scalar_cols].to_numpy(dtype=np.float64))
        X_out = X_out[self.gi_cols + self.mou_cols + self.scalar_cols]

        if y is None:
            return X_out, None

        y_arr = self._transform_gi_block(y.to_numpy(dtype=np.float64))
        y_out = pd.DataFrame(y_arr, index=y.index, columns=self.y_cols)
        return X_out, y_out

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        arr = y_scaled * self.gi_sigma + self.gi_mu
        arr = np.expm1(arr)
        return np.clip(arr, a_min=0.0, a_max=None)


class PartsDataset(Dataset):
    def __init__(self, X, y, gi_cols, mou_cols, scalar_cols):
        self.gi = X[gi_cols].to_numpy(dtype=np.float32)
        self.mou = X[mou_cols].to_numpy(dtype=np.float32) if mou_cols else np.zeros((len(X), 0), dtype=np.float32)
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
    def __init__(self, n_scalars, hidden=64, dropout=0.2, n_outputs=6, use_mou=True):
        super().__init__()
        self.use_mou = use_mou
        self.gi_lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        if use_mou:
            self.mou_lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
            fusion_in = 2 * hidden + n_scalars
        else:
            self.mou_lstm = None
            fusion_in = hidden + n_scalars
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
        parts = [h_gi[-1]]
        if self.use_mou and mou_seq.shape[1] > 0:
            _, (h_mou, _) = self.mou_lstm(mou_seq)
            parts.append(h_mou[-1])
        parts.append(scalars)
        feat = torch.cat(parts, dim=-1)
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


def chronological_split_by_version(X_train: pd.DataFrame, y_train: pd.DataFrame, val_frac: float = 0.10):
    """Last `val_frac` of distinct versions go to val. Falls back to row-wise if only 1 version."""
    X_train = X_train.sort_index(level='version')
    y_train = y_train.sort_index(level='version')
    versions_sorted = sorted(pd.unique(X_train.index.get_level_values('version')))
    if len(versions_sorted) <= 1:
        print('Warning: only 1 distinct version; falling back to row-wise 90/10 split.')
        n = len(X_train)
        n_val = max(1, int(round(val_frac * n)))
        return X_train.iloc[:-n_val], X_train.iloc[-n_val:], y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    n_val = max(1, int(round(val_frac * len(versions_sorted))))
    val_versions = set(versions_sorted[-n_val:])
    is_val = X_train.index.get_level_values('version').isin(val_versions)
    return X_train[~is_val], X_train[is_val], y_train[~is_val], y_train[is_val]


def _train_one_config(
    combo: Dict,
    X_tr, y_tr, X_val, y_val,
    gi_cols, mou_cols, scalar_cols,
    n_outputs, fixed,
):
    """Fit preprocessor + model for a single HP combo. Returns (best_val_score, best_epoch, preproc, state_dict)."""
    preproc = DLPreprocessor().fit(X_tr, y_tr, gi_cols, mou_cols, scalar_cols)
    X_tr_s, y_tr_s = preproc.transform(X_tr, y_tr)
    X_val_s, _ = preproc.transform(X_val, None)

    train_ds = PartsDataset(X_tr_s, y_tr_s, gi_cols, mou_cols, scalar_cols)
    val_ds = PartsDataset(X_val_s, None, gi_cols, mou_cols, scalar_cols)

    weights = make_recency_weights(X_tr, fixed['recency_min_weight'], fixed['recency_max_weight'])
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=fixed['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = DualEncoderLSTM(
        n_scalars=len(scalar_cols),
        hidden=combo['hidden'],
        dropout=combo['dropout'],
        n_outputs=n_outputs,
        use_mou=bool(mou_cols),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=combo['lr'], weight_decay=fixed['weight_decay'])
    loss_fn = nn.MSELoss()

    best_val = -math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    for epoch in range(fixed['epochs']):
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

        val_preds_scaled = predict_all(model, val_loader, has_y=False)
        val_preds = preproc.inverse_y(val_preds_scaled)
        val_score = get_user_defined_score(y_val.values, val_preds)
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= fixed['patience']:
                break

    return best_val, best_epoch, preproc, best_state


def DLModel(
    df_gi: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_date: str,
    validation_columns: List[str],
    do_hp_tuning: bool = False,
    hp_grid: Optional[Dict[str, List]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str]], float, float]:
    """
    Deep-learning drop-in for MLModel (v2).

    Caller should pass unscaled X_train / y_train / X_test (split_data(..., scale=False)).
    DLModel fits its own preprocessor internally.

    do_hp_tuning=False: single default config (hidden=64, dropout=0.2, lr=1e-3).
    do_hp_tuning=True: small grid (DEFAULT_HP_GRID, 8 combos) — overridable via hp_grid.

    Flow: 90/10 chronological split by version -> train config(s) -> pick best by val score
          -> refit preproc on full X_train/y_train -> final train on 100% for best_epoch
          -> inverse-transform test predictions -> post-processing (verbatim from MLModel).
    """

    # --- 1. Feature groups ---
    gi_cols, mou_cols, scalar_cols = identify_feature_groups(list(X_train.columns))
    n_outputs = y_train.shape[1]

    fixed = {
        'batch_size': 64,
        'epochs': 50,
        'patience': 8,
        'weight_decay': 1e-5,
        'optimizer': 'Adam',
        'loss': 'MSE',
        'recency_min_weight': 1.0,
        'recency_max_weight': 20.0,
    }
    default_combo = {'hidden': 64, 'dropout': 0.2, 'lr': 1e-3}

    # --- 2. 90/10 chronological split ---
    X_tr, X_val, y_tr, y_val = chronological_split_by_version(X_train, y_train, val_frac=0.10)
    print(f'Chronological split: train={len(X_tr)}, val={len(X_val)}')

    # --- 3. HP search or single-config training ---
    print('-' * 15, 'DL Training', '-' * 15)
    if do_hp_tuning:
        grid = hp_grid if hp_grid is not None else DEFAULT_HP_GRID
        keys = list(grid.keys())
        combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]
        print(f'HP tuning over {len(combos)} combos.')
        hp_log = []
        best = None  # (val_score, combo, best_epoch, preproc, state)
        for combo in combos:
            val_score, best_epoch, preproc, state = _train_one_config(
                combo, X_tr, y_tr, X_val, y_val, gi_cols, mou_cols, scalar_cols, n_outputs, fixed,
            )
            hp_log.append({'combo': combo, 'val_score': float(val_score), 'best_epoch': int(best_epoch)})
            print(f'  combo={combo} val_score={val_score:.4f} best_epoch={best_epoch}')
            if best is None or val_score > best[0]:
                best = (val_score, combo, best_epoch, preproc, state)
        _vs, winning_combo, winning_epoch, preproc_90, state_90 = best
    else:
        winning_combo = default_combo
        hp_log = None
        _vs, winning_epoch, preproc_90, state_90 = _train_one_config(
            winning_combo, X_tr, y_tr, X_val, y_val, gi_cols, mou_cols, scalar_cols, n_outputs, fixed,
        )

    # --- 4. Compute cv_train / cv_validation with winning combo on the 90/10 split ---
    model_90 = DualEncoderLSTM(
        n_scalars=len(scalar_cols),
        hidden=winning_combo['hidden'],
        dropout=winning_combo['dropout'],
        n_outputs=n_outputs,
        use_mou=bool(mou_cols),
    ).to(device)
    model_90.load_state_dict(state_90)

    X_tr_s, _ = preproc_90.transform(X_tr, None)
    X_val_s, _ = preproc_90.transform(X_val, None)
    tr_loader_eval = DataLoader(PartsDataset(X_tr_s, None, gi_cols, mou_cols, scalar_cols), batch_size=256, shuffle=False)
    val_loader_eval = DataLoader(PartsDataset(X_val_s, None, gi_cols, mou_cols, scalar_cols), batch_size=256, shuffle=False)

    tr_preds = preproc_90.inverse_y(predict_all(model_90, tr_loader_eval, has_y=False))
    val_preds = preproc_90.inverse_y(predict_all(model_90, val_loader_eval, has_y=False))
    cv_train_score = float(get_user_defined_score(y_tr.values, tr_preds))
    cv_validation_score = float(get_user_defined_score(y_val.values, val_preds))
    if winning_epoch <= 0:
        winning_epoch = fixed['epochs']

    # --- 5. Final retrain on 100% of training data with winning combo for winning_epoch epochs ---
    print(f'Final retrain on 100% of X_train for {winning_epoch} epochs (combo={winning_combo}).')
    preproc_full = DLPreprocessor().fit(X_train, y_train, gi_cols, mou_cols, scalar_cols)
    X_train_s, y_train_s = preproc_full.transform(X_train, y_train)
    X_test_s, _ = preproc_full.transform(X_test, None)

    train_ds = PartsDataset(X_train_s, y_train_s, gi_cols, mou_cols, scalar_cols)
    test_ds = PartsDataset(X_test_s, None, gi_cols, mou_cols, scalar_cols)
    weights = make_recency_weights(X_train, fixed['recency_min_weight'], fixed['recency_max_weight'])
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=fixed['batch_size'], sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model_full = DualEncoderLSTM(
        n_scalars=len(scalar_cols),
        hidden=winning_combo['hidden'],
        dropout=winning_combo['dropout'],
        n_outputs=n_outputs,
        use_mou=bool(mou_cols),
    ).to(device)
    optimizer = torch.optim.Adam(model_full.parameters(), lr=winning_combo['lr'], weight_decay=fixed['weight_decay'])
    loss_fn = nn.MSELoss()

    for epoch in range(winning_epoch):
        model_full.train()
        for gi, mou, scal, y in train_loader:
            gi = gi.to(device)
            mou = mou.to(device)
            scal = scal.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model_full(gi, mou, scal), y)
            loss.backward()
            optimizer.step()

    # --- 6. Predict X_test (scaled) and inverse-transform ---
    test_preds_scaled = predict_all(model_full, test_loader, has_y=False)
    test_preds = preproc_full.inverse_y(test_preds_scaled)

    start_date_ts = pd.to_datetime(test_date, format=month_format)
    prediction_dates = pd.date_range(
        start=start_date_ts,
        periods=len(validation_columns),
        freq='MS',
    ).strftime(month_format)

    pred_df = pd.DataFrame(test_preds, index=X_test.index, columns=prediction_dates)
    if not isinstance(pred_df.index, pd.MultiIndex) or pred_df.index.names != ['part', 'plant', 'version']:
        pred_df.index = pd.MultiIndex.from_tuples(pred_df.index, names=['part', 'plant', 'version'])

    # --- 7. Post-processing (copied verbatim from MLModel) ---
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

    # --- 8. Build grid_params return ---
    grid_params: Dict[str, Union[int, float, str]] = {
        'architecture': 'DualEncoderLSTM',
        'hidden': winning_combo['hidden'],
        'dropout': winning_combo['dropout'],
        'lr': winning_combo['lr'],
        'weight_decay': fixed['weight_decay'],
        'batch_size': fixed['batch_size'],
        'epochs_trained': winning_epoch,
        'patience': fixed['patience'],
        'loss': fixed['loss'],
        'optimizer': fixed['optimizer'],
        'recency_min_weight': fixed['recency_min_weight'],
        'recency_max_weight': fixed['recency_max_weight'],
        'hp_tuning': do_hp_tuning,
    }
    if do_hp_tuning:
        grid_params['hp_grid_log'] = hp_log

    return pred_df_final, grid_params, cv_train_score, cv_validation_score
