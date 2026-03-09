"""
ToTo deep learning with:
1) automated hyperparameter tuning (walk-forward validation),
2) expanding-window historical backtest,
3) one single self-contained HTML report (all visuals + tables + metrics).
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CUDA async allocator tends to improve GPU memory reuse and throughput on TF 2.x.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def ensure_cuda_dll_paths() -> None:
    """
    Make pip-installed CUDA DLL directories visible to TensorFlow on Windows.
    This avoids missing cudart/cublas/cudnn DLL errors when using wheels that
    bundle CUDA libs under site-packages/nvidia/*/bin.
    """
    if os.name != "nt":
        return

    submods = ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse"]
    root_candidates: List[Path] = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
        Path(__file__).resolve().parent / ".venv_tf_cuda" / "Lib" / "site-packages" / "nvidia",
    ]

    dll_dirs: List[str] = []
    seen: set[str] = set()
    for root in root_candidates:
        if not root.exists():
            continue
        for sub in submods:
            p = root / sub / "bin"
            if p.is_dir():
                s = str(p.resolve())
                if s not in seen:
                    seen.add(s)
                    dll_dirs.append(s)

    if not dll_dirs:
        return

    for d in dll_dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

    current_path = os.environ.get("PATH", "")
    prepend = [d for d in dll_dirs if d not in current_path]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + ([current_path] if current_path else []))


ensure_cuda_dll_paths()

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
ACTIVE_SEED = SEED


def set_global_seed(seed: int) -> None:
    global ACTIVE_SEED
    ACTIVE_SEED = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(SEED)

WIN_COLS = ["Win_1", "Win_2", "Win_3", "Win_4", "Win_5", "Win_6"]
PRIMARY_COLS = WIN_COLS + ["Addl No."]

BLEND_WEIGHTS = {
    "model_soft": 0.9586,
    "model_set": 0.3586,
    "cluster": 0.4802,
    "repel": 0.1677,
    "addl_model": 0.3917,
    "addl_cluster": 0.3867,
}

WIN_BLEND_KEYS = ["model_soft", "model_set", "cluster"]
ADDL_BLEND_KEYS = ["addl_model", "addl_cluster"]

DEFAULT_FEATURE_GROUP_WEIGHTS = {
    "primary": 4.0,
    "other": 1.25,
    "repel": 1.20,
    "cluster": 1.35,
    "line": 1.60,
}


@dataclass
class ModelConfig:
    seq_len: int
    lstm_units: int
    gru_units: int
    dense_units: int
    dropout: float
    lr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ToTo Predication Report"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Global random seed")
    parser.add_argument("--sweep-mode", action="store_true", help="Use lighter settings for automated multi-run sweeps")
    parser.add_argument("--csv", default="ToTo-05_Mar_2026.csv", help="Input CSV file path")
    parser.add_argument("--clusters", type=int, default=0, help="Force KMeans clusters (0=auto)")
    parser.add_argument("--tune-trials", type=int, default=14, help="Random trials sampled from config space")
    parser.add_argument("--tune-folds", type=int, default=3, help="Walk-forward folds per tuning trial")
    parser.add_argument("--tune-epochs", type=int, default=10, help="Epoch cap for each tuning fold")
    parser.add_argument("--multi-restarts", type=int, default=5, help="Number of final-training restarts (best restart kept by hit objective)")
    parser.add_argument("--final-epochs", type=int, default=64, help="Final training epoch cap")
    parser.add_argument("--backtest-folds", type=int, default=10, help="Expanding-window backtest folds")
    parser.add_argument("--backtest-epochs", type=int, default=20, help="Epoch cap per backtest fold")
    parser.add_argument(
        "--focus-last-n",
        type=int,
        default=10,
        help="Optimize and backtest only the most recent N draws (0 disables recent-focus mode)",
    )
    parser.add_argument("--output-html", default="", help="Output HTML path (optional)")
    parser.add_argument("--w-primary", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["primary"], help="Feature weight for Win_1..Win_6 + Addl No.")
    parser.add_argument("--w-other", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["other"], help="Feature weight for non-primary scalar columns")
    parser.add_argument("--w-repel", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["repel"], help="Feature weight for repel features")
    parser.add_argument("--w-cluster", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["cluster"], help="Feature weight for cluster-prior features")
    parser.add_argument("--w-line", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["line"], help="Feature weight for line-endpoint convergence features")
    parser.add_argument(
        "--perf-mode",
        choices=["auto", "high"],
        default="high",
        help="Hardware/runtime tuning mode (high tries larger batches and faster execution settings)",
    )
    parser.add_argument(
        "--gpu-preallocate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve near-full GPU memory instead of growth mode",
    )
    parser.add_argument("--gpu-batch-size", type=int, default=320, help="Override model training batch size")
    parser.add_argument("--diffusion-batch-size", type=int, default=128, help="Override diffusion training batch size")
    parser.add_argument("--steps-per-execution", type=int, default=32, help="Keras steps_per_execution (0 = auto)")
    parser.add_argument(
        "--dataset-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache tf.data datasets in RAM to reduce input overhead",
    )
    parser.add_argument(
        "--train-recency-weighted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use recency-weighted sample weights during model training",
    )
    parser.add_argument("--reward-epochs", type=int, default=10, help="Reward-guided self-improvement epochs after final training")
    parser.add_argument("--reward-window", type=int, default=320, help="Recent sequence samples used for reward refinement")
    parser.add_argument("--reward-min-samples", type=int, default=64, help="Minimum reward refinement sample count")
    parser.add_argument(
        "--hit-score-focused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-increase search depth and recent-window emphasis for hit-score optimization",
    )
    parser.add_argument("--blend-random-candidates", type=int, default=3600, help="Random blend candidates evaluated before coordinate-ascent refinement")
    parser.add_argument("--blend-coordinate-iters", type=int, default=14, help="Coordinate-ascent iterations after random blend search")
    parser.add_argument("--blend-coordinate-step", type=float, default=0.42, help="Initial coordinate-ascent relative step size")
    parser.add_argument("--blend-simplex-iters", type=int, default=160, help="Local Nelder-Mead iterations after coordinate-ascent")
    parser.add_argument("--blend-simplex-step", type=float, default=0.24, help="Initial simplex step size in transformed blend space")
    parser.add_argument("--blend-tail-iters", type=int, default=8, help="Tail-hit local refinement iterations after Nelder-Mead")
    parser.add_argument("--blend-tail-candidates", type=int, default=360, help="Candidates evaluated per tail-hit refinement iteration")
    parser.add_argument("--backtest-local-random-candidates", type=int, default=2600, help="Local blend random candidates used inside strict walk-forward backtest")
    parser.add_argument("--backtest-restarts", type=int, default=1, help="Model restarts per walk-forward step (recent-focus mode)")
    parser.add_argument("--restart-ensemble-topk", type=int, default=1, help="Average top-K restarts per walk-forward step")
    parser.add_argument(
        "--backtest-optimize-avg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize strict walk-forward local blend for avg_win_hits emphasis",
    )
    parser.add_argument(
        "--feature-weight-trials",
        type=int,
        default=0,
        help="Random-search trials for feature group weights (0 disables)",
    )
    parser.add_argument(
        "--feature-weight-epochs",
        type=int,
        default=6,
        help="Epoch cap used during feature-weight tuning proxy fits",
    )
    parser.add_argument(
        "--diffusion-window",
        type=int,
        default=180,
        help="Recent rows used for diffusion pattern image (clamped to available rows)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=72,
        help="DDPM denoising steps",
    )
    parser.add_argument(
        "--diffusion-epochs",
        type=int,
        default=14,
        help="Epochs per diffusion trial",
    )
    parser.add_argument(
        "--diffusion-trials",
        type=int,
        default=5,
        help="Number of diffusion hyperparameter trials",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=12,
        help="Images sampled per diffusion trial for score selection",
    )
    parser.add_argument(
        "--diffusion-future-samples",
        type=int,
        default=240,
        help="Synthetic windows used for diffusion next-day prediction",
    )
    return parser.parse_args()


def configure_hardware(
    preallocate_gpu: bool = False,
    gpu_batch_size_override: int = 0,
    perf_mode: str = "auto",
) -> Dict[str, object]:
    hw: Dict[str, object] = {
        "gpu_count": 0,
        "gpu_names": [],
        "mixed_precision": False,
        "xla": False,
        "batch_size": 64,
    }
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return hw

    if not preallocate_gpu:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    hw["gpu_count"] = len(gpus)
    hw["gpu_names"] = [g.name for g in gpus]
    # In preallocate mode we start with a larger batch to drive higher VRAM occupancy.
    if preallocate_gpu:
        hw["batch_size"] = 640 if str(perf_mode).lower() == "high" else 512
    else:
        hw["batch_size"] = 320 if str(perf_mode).lower() == "high" else 256
    if gpu_batch_size_override > 0:
        hw["batch_size"] = int(gpu_batch_size_override)

    disable_xla = os.getenv("TOTO_DISABLE_XLA", "").strip().lower() in {"1", "true", "yes", "y"}
    has_ptxas = shutil.which("ptxas.exe") is not None
    if disable_xla or not has_ptxas:
        hw["xla"] = False
    else:
        try:
            tf.config.optimizer.set_jit(True)
            hw["xla"] = True
        except Exception:
            hw["xla"] = False

    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        hw["mixed_precision"] = True
    except Exception:
        hw["mixed_precision"] = False
    return hw


def parse_ratio_column(series: pd.Series, default_left: int = 3, default_right: int = 3) -> Tuple[pd.Series, pd.Series]:
    extracted = series.fillna("").astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
    left = pd.to_numeric(extracted[0], errors="coerce").fillna(default_left).astype(int)
    right = pd.to_numeric(extracted[1], errors="coerce").fillna(default_right).astype(int)
    return left, right


def parse_from_last_count(value: object) -> int:
    if pd.isna(value):
        return 0
    return len(re.findall(r"\d+", str(value)))


def parse_from_last_mean(value: object) -> float:
    if pd.isna(value):
        return 0.0
    nums = [int(x) for x in re.findall(r"\d+", str(value))]
    return float(np.mean(nums)) if nums else 0.0


def prepare_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    required = set(PRIMARY_COLS + ["Draw", "Date", "Low/High", "Odd/Even"])
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_raw.copy()
    df["Draw"] = pd.to_numeric(df["Draw"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Draw", "Date"], ascending=[True, True], na_position="last").reset_index(drop=True)

    low_count, high_count = parse_ratio_column(df["Low/High"], 3, 3)
    odd_count, even_count = parse_ratio_column(df["Odd/Even"], 3, 3)
    df["Low_Count"] = low_count
    df["High_Count"] = high_count
    df["Odd_Count"] = odd_count
    df["Even_Count"] = even_count
    df["From_Last_Count"] = df.get("From Last", pd.Series([np.nan] * len(df))).apply(parse_from_last_count)
    df["From_Last_Mean"] = df.get("From Last", pd.Series([np.nan] * len(df))).apply(parse_from_last_mean)

    other_cols = [
        "Sum",
        "Average",
        "1-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "Low_Count",
        "High_Count",
        "Odd_Count",
        "Even_Count",
        "From_Last_Count",
        "From_Last_Mean",
    ]

    for col in PRIMARY_COLS + other_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df, PRIMARY_COLS, other_cols


def build_role_encoded_grid(df: pd.DataFrame) -> np.ndarray:
    """
    Encode Win_1..Win_6 + Addl No. into a draw x number grid.
    Values are role-intensity encoded for diffusion image learning.
    """
    n = len(df)
    grid = np.zeros((n, 49), dtype=np.float32)
    row_idx = np.arange(n)

    for role_i, col in enumerate(WIN_COLS, start=1):
        nums = np.clip(df[col].astype(int).values, 1, 49) - 1
        grid[row_idx, nums] = role_i / 7.5

    addl_nums = np.clip(df["Addl No."].astype(int).values, 1, 49) - 1
    grid[row_idx, addl_nums] = 1.0
    return grid


def build_binary_draw_grid(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    grid = np.zeros((n, 49), dtype=np.float32)
    row_idx = np.arange(n)
    for col in PRIMARY_COLS:
        nums = np.clip(df[col].astype(int).values, 1, 49) - 1
        grid[row_idx, nums] = 1.0
    return grid


def build_diffusion_windows(role_grid: np.ndarray, window: int) -> np.ndarray:
    if window < 24:
        raise ValueError("diffusion_window must be >= 24")
    if len(role_grid) < window:
        raise ValueError(f"Not enough rows ({len(role_grid)}) for diffusion window={window}")
    windows = np.stack([role_grid[end - window : end] for end in range(window, len(role_grid) + 1)], axis=0)
    return windows[..., None].astype(np.float32)


def diffusion_schedule(num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    betas = np.linspace(1e-4, 0.02, num_steps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


def build_diffusion_denoiser(height: int, width: int, base_filters: int = 32, dropout: float = 0.10) -> keras.Model:
    noisy_inp = keras.Input(shape=(height, width, 1), name="noisy_grid")
    t_inp = keras.Input(shape=(1,), name="step_norm")

    t = layers.Dense(base_filters, activation="swish")(t_inp)
    t = layers.Dense(base_filters, activation="swish")(t)
    t = layers.Reshape((1, 1, base_filters))(t)
    t = layers.Lambda(lambda z: tf.tile(z, [1, height, width, 1]), name="time_broadcast")(t)

    x = layers.Concatenate(name="x_plus_t")([noisy_inp, t])
    x = layers.Conv2D(base_filters, 3, padding="same", name="entry_conv")(x)
    x = layers.Activation("swish")(x)

    for block_i in range(4):
        res = x
        x = layers.Conv2D(base_filters, 3, padding="same", name=f"res_{block_i}_conv1")(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(dropout, name=f"res_{block_i}_drop")(x)
        x = layers.Conv2D(base_filters, 3, padding="same", name=f"res_{block_i}_conv2")(x)
        x = layers.Add(name=f"res_{block_i}_add")([x, res])
        x = layers.Activation("swish")(x)

    x = layers.Conv2D(max(16, base_filters // 2), 3, padding="same", activation="swish", name="head_conv")(x)
    out = layers.Conv2D(1, 1, padding="same", dtype="float32", name="noise_pred")(x)

    model = keras.Model(inputs=[noisy_inp, t_inp], outputs=out, name="ToTo_Diffusion_Denoiser")
    return model


def train_diffusion_denoiser(
    x0_windows: np.ndarray,
    num_steps: int,
    epochs: int,
    batch_size: int,
    base_filters: int,
    lr: float,
) -> Tuple[keras.Model, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    tf.keras.backend.clear_session()
    h, w = x0_windows.shape[1], x0_windows.shape[2]
    model = build_diffusion_denoiser(height=h, width=w, base_filters=base_filters)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss="mse")

    betas, alphas, alpha_bars = diffusion_schedule(num_steps)
    losses: List[float] = []
    n = len(x0_windows)
    for epoch in range(epochs):
        order = np.random.permutation(n)
        batch_losses: List[float] = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x0 = x0_windows[idx]
            b = len(idx)
            if b == 0:
                continue
            t = np.random.randint(0, num_steps, size=b, dtype=np.int32)
            noise = np.random.normal(size=x0.shape).astype(np.float32)
            a_bar = alpha_bars[t].reshape(-1, 1, 1, 1)
            noisy = np.sqrt(a_bar).astype(np.float32) * x0 + np.sqrt(1.0 - a_bar).astype(np.float32) * noise
            t_norm = (t.astype(np.float32) / max(1.0, float(num_steps - 1))).reshape(-1, 1)
            loss = float(model.train_on_batch([noisy, t_norm], noise))
            batch_losses.append(loss)
        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        losses.append(epoch_loss)
    return model, betas, alphas, alpha_bars, losses


def sample_ddpm(
    model: keras.Model,
    sample_count: int,
    height: int,
    width: int,
    betas: np.ndarray,
    alphas: np.ndarray,
    alpha_bars: np.ndarray,
) -> np.ndarray:
    num_steps = len(betas)
    x = np.random.normal(size=(sample_count, height, width, 1)).astype(np.float32)

    for t in range(num_steps - 1, -1, -1):
        t_norm = np.full((sample_count, 1), t / max(1.0, float(num_steps - 1)), dtype=np.float32)
        eps_pred = model.predict([x, t_norm], verbose=0, batch_size=min(64, sample_count))
        alpha_t = float(alphas[t])
        alpha_bar_t = float(alpha_bars[t])
        beta_t = float(betas[t])
        coef1 = 1.0 / math.sqrt(max(alpha_t, 1e-8))
        coef2 = (1.0 - alpha_t) / math.sqrt(max(1.0 - alpha_bar_t, 1e-8))
        x = coef1 * (x - coef2 * eps_pred)
        if t > 0:
            noise = np.random.normal(size=x.shape).astype(np.float32)
            x = x + math.sqrt(max(beta_t, 1e-8)) * noise

    return np.clip((x + 1.0) * 0.5, 0.0, 1.0)[..., 0]


def topk_binary_rows(matrix: np.ndarray, k: int = 7) -> np.ndarray:
    m = matrix.copy()
    idx = np.argpartition(m, -k, axis=1)[:, -k:]
    out = np.zeros_like(m, dtype=np.float32)
    rows = np.arange(m.shape[0])[:, None]
    out[rows, idx] = 1.0
    return out


def evaluate_pattern_match(generated: np.ndarray, target_role: np.ndarray, target_binary: np.ndarray) -> Dict[str, object]:
    generated = np.clip(generated.astype(np.float32), 0.0, 1.0)
    gen_binary = topk_binary_rows(generated, k=7)
    overlap = gen_binary * target_binary
    hit_rate = float(overlap.sum() / max(1.0, target_binary.sum()))
    mse = float(np.mean((generated - target_role) ** 2))
    acc_map = 1.0 - np.abs(generated - target_role)
    row_hits = overlap.sum(axis=1).astype(float)
    row_hit_ge3 = float(np.mean(row_hits >= 3.0))
    row_hit_ge4 = float(np.mean(row_hits >= 4.0))

    target_mass = float(np.sum(target_role)) + 1e-8
    role_match = float(np.sum(generated * target_role) / target_mass)

    gen_profile = generated.mean(axis=0).astype(np.float64)
    tgt_profile = target_role.mean(axis=0).astype(np.float64)
    profile_den = (np.linalg.norm(gen_profile) * np.linalg.norm(tgt_profile)) + 1e-8
    profile_cos = float(np.dot(gen_profile, tgt_profile) / profile_den)
    profile_cos = float(np.clip(profile_cos, 0.0, 1.0))

    row_num = np.sum(generated * target_role, axis=1).astype(np.float64)
    row_den = (np.linalg.norm(generated, axis=1) * np.linalg.norm(target_role, axis=1)) + 1e-8
    row_cos = np.clip(row_num / row_den, 0.0, 1.0)
    row_cos_mean = float(np.mean(row_cos))

    score = (
        0.36 * hit_rate
        + 0.16 * row_hit_ge3
        + 0.08 * row_hit_ge4
        + 0.15 * role_match
        + 0.13 * profile_cos
        + 0.08 * row_cos_mean
        + 0.04 * (1.0 - mse)
    )
    score = float(np.clip(score, 0.0, 1.0))
    return {
        "score": float(score),
        "hit_rate": hit_rate,
        "mse": mse,
        "row_hit_ge3": row_hit_ge3,
        "row_hit_ge4": row_hit_ge4,
        "role_match": role_match,
        "profile_cos": profile_cos,
        "row_cos": row_cos_mean,
        "generated_binary": gen_binary,
        "overlap_map": overlap,
        "accuracy_map": np.clip(acc_map, 0.0, 1.0).astype(np.float32),
        "row_hits": row_hits,
    }


def diffusion_predict_next_day(
    samples: np.ndarray,
    actual_last_row: np.ndarray,
    recent_binary: np.ndarray | None = None,
    sample_limit: int = 160,
) -> Tuple[List[int], int, np.ndarray, np.ndarray]:
    if len(samples) == 0:
        uniform = np.ones(49, dtype=np.float32) / 49.0
        return [1, 2, 3, 4, 5, 6], 7, uniform, uniform

    sample_limit = max(8, min(sample_limit, len(samples)))
    selected = samples[:sample_limit]
    win_scores = np.zeros(49, dtype=np.float64)
    addl_scores = np.zeros(49, dtype=np.float64)
    endpoint_scores = np.zeros(49, dtype=np.float64)
    next_row_mean = np.zeros(49, dtype=np.float64)
    total_weight = 1e-8

    actual_vec = actual_last_row.astype(np.float64)
    actual_norm = np.linalg.norm(actual_vec) + 1e-8

    if recent_binary is not None and len(recent_binary) > 0:
        hist_prior = normalize_prob(np.mean(recent_binary.astype(np.float64), axis=0))
    else:
        hist_prior = np.ones(49, dtype=np.float64) / 49.0

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

    for window in selected:
        next_row = window[-1].astype(np.float64)
        prev_row = window[-2].astype(np.float64) if window.shape[0] > 1 else next_row
        prev2_row = window[-3].astype(np.float64) if window.shape[0] > 2 else prev_row

        sim_prev = max(0.0, _cosine(prev_row, actual_vec))
        trend_row = 0.65 * prev_row + 0.35 * prev2_row
        sim_trend = max(0.0, float(np.dot(trend_row, actual_vec) / ((np.linalg.norm(trend_row) + 1e-8) * actual_norm)))
        temporal_consistency = max(0.0, _cosine(prev_row, prev2_row))

        row_prob = normalize_prob(next_row + 1e-6)
        entropy = float(-np.sum(row_prob * np.log(row_prob + 1e-12)) / np.log(49.0))
        confidence = float(np.max(row_prob))
        stability = float(np.clip(1.0 - 0.75 * entropy, 0.12, 1.0))
        weight = 0.45 * sim_prev + 0.24 * sim_trend + 0.16 * temporal_consistency + 0.15 * confidence
        weight = max(0.03, weight * stability)

        blended_row = normalize_prob(0.74 * row_prob + 0.26 * hist_prior)

        top7 = np.argsort(blended_row)[-7:][::-1]
        win_idx = top7[:6]
        addl_idx = int(top7[6])
        win_scores[win_idx] += weight * blended_row[win_idx]
        addl_scores[addl_idx] += weight * blended_row[addl_idx]
        endpoint_scores[top7] += weight
        next_row_mean += weight * blended_row
        total_weight += weight

    next_row_mean /= total_weight
    win_prob = normalize_prob(
        0.60 * normalize_prob(win_scores)
        + 0.24 * normalize_prob(endpoint_scores)
        + 0.16 * normalize_prob(next_row_mean + hist_prior)
    )
    addl_prob = normalize_prob(0.74 * normalize_prob(addl_scores) + 0.26 * win_prob)

    pred_win_idx = np.argsort(win_prob)[-6:]
    pred_win = sorted([int(i + 1) for i in pred_win_idx.tolist()])
    blocked = np.zeros(49, dtype=bool)
    blocked[pred_win_idx] = True
    addl_work = addl_prob.copy()
    addl_work[blocked] = 0.0
    if float(addl_work.sum()) <= 0.0:
        addl_work = addl_prob.copy()
        addl_work[blocked] = 1e-12
    pred_addl = int(np.argmax(addl_work)) + 1
    return pred_win, pred_addl, win_prob.astype(np.float32), next_row_mean.astype(np.float32)


def run_diffusion_suite(
    df: pd.DataFrame,
    window: int,
    num_steps: int,
    epochs: int,
    trials: int,
    samples_per_trial: int,
    future_samples: int,
    gpu_batch_size: int,
    diffusion_batch_size: int = 0,
) -> Dict[str, object]:
    role_grid = build_role_encoded_grid(df)
    binary_grid = build_binary_draw_grid(df)
    window = int(max(24, min(window, len(df))))

    x_windows = build_diffusion_windows(role_grid, window)
    x0_windows = (x_windows * 2.0 - 1.0).astype(np.float32)
    target_role = role_grid[-window:].astype(np.float32)
    target_binary = binary_grid[-window:].astype(np.float32)

    trial_space = [
        {"base_filters": 24, "lr": 1.0e-3, "epochs": max(6, epochs - 2)},
        {"base_filters": 32, "lr": 8.0e-4, "epochs": epochs},
        {"base_filters": 40, "lr": 6.0e-4, "epochs": epochs + 2},
        {"base_filters": 28, "lr": 7.0e-4, "epochs": epochs + 1},
        {"base_filters": 36, "lr": 5.5e-4, "epochs": epochs + 2},
        {"base_filters": 40, "lr": 4.8e-4, "epochs": epochs + 3},
    ]
    trial_space = trial_space[: max(1, trials)]

    if diffusion_batch_size > 0:
        diff_batch = int(max(8, diffusion_batch_size))
    else:
        # Let diffusion consume a larger batch window on 6GB-class GPUs.
        diff_batch = int(max(32, min(320, gpu_batch_size)))

    def _train_diffusion_with_fallback(cfg: Dict[str, float], start_batch: int) -> Tuple[keras.Model, np.ndarray, np.ndarray, np.ndarray, List[float], int]:
        local_batch = int(max(8, start_batch))
        while True:
            try:
                model, betas, alphas, alpha_bars, losses = train_diffusion_denoiser(
                    x0_windows=x0_windows,
                    num_steps=num_steps,
                    epochs=int(cfg["epochs"]),
                    batch_size=local_batch,
                    base_filters=int(cfg["base_filters"]),
                    lr=float(cfg["lr"]),
                )
                return model, betas, alphas, alpha_bars, losses, local_batch
            except tf.errors.ResourceExhaustedError:
                if local_batch <= 16:
                    raise
                next_batch = max(16, local_batch // 2)
                print(f"[DIFFUSION] OOM at batch={local_batch}; retrying with batch={next_batch}")
                local_batch = int(next_batch)

    trial_rows: List[Dict[str, object]] = []
    best_visual_payload: Dict[str, object] | None = None
    best_cfg: Dict[str, float] | None = None
    best_batch = diff_batch
    best_score = -1.0

    for trial_id, cfg in enumerate(trial_space, start=1):
        model, betas, alphas, alpha_bars, losses, trial_batch = _train_diffusion_with_fallback(cfg, diff_batch)
        diff_batch = int(min(diff_batch, trial_batch))

        sampled = sample_ddpm(
            model=model,
            sample_count=max(2, samples_per_trial),
            height=window,
            width=49,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
        )

        sample_scores: List[float] = []
        sample_metrics: List[Dict[str, object]] = []
        for s in sampled:
            m = evaluate_pattern_match(s, target_role=target_role, target_binary=target_binary)
            sample_scores.append(float(m["score"]))
            sample_metrics.append(m)

        rank_idx = np.argsort(np.asarray(sample_scores, dtype=np.float64))[::-1]
        best_sample_i = int(rank_idx[0])
        best_sample = sampled[best_sample_i]
        best_sample_metrics = sample_metrics[best_sample_i]

        top_k = int(min(4, len(rank_idx)))
        top_idx = rank_idx[:top_k]
        consensus_sample = np.mean(sampled[top_idx], axis=0)
        consensus_metrics = evaluate_pattern_match(consensus_sample, target_role=target_role, target_binary=target_binary)
        trial_score = float(0.66 * consensus_metrics["score"] + 0.34 * best_sample_metrics["score"])
        display_sample = consensus_sample if float(consensus_metrics["score"]) >= float(best_sample_metrics["score"]) else best_sample
        display_metrics = consensus_metrics if float(consensus_metrics["score"]) >= float(best_sample_metrics["score"]) else best_sample_metrics
        trial_rows.append(
            {
                "trial": trial_id,
                "base_filters": int(cfg["base_filters"]),
                "lr": float(cfg["lr"]),
                "epochs": int(cfg["epochs"]),
                "steps": int(num_steps),
                "batch_size": int(trial_batch),
                "best_score": trial_score,
                "best_hit_rate": float(best_sample_metrics["hit_rate"]),
                "consensus_score": float(consensus_metrics["score"]),
                "consensus_hit_rate": float(consensus_metrics["hit_rate"]),
                "consensus_row_ge3": float(consensus_metrics["row_hit_ge3"]),
                "consensus_profile_cos": float(consensus_metrics["profile_cos"]),
                "best_mse": float(best_sample_metrics["mse"]),
                "final_train_loss": float(losses[-1]) if losses else float("nan"),
            }
        )
        print(
            f"[DIFFUSION] Trial {trial_id}/{len(trial_space)} "
            f"filters={cfg['base_filters']} lr={cfg['lr']:.6f} epochs={cfg['epochs']} "
            f"score={trial_score:.4f} consensus={consensus_metrics['score']:.4f} "
            f"hit={best_sample_metrics['hit_rate']:.4f} mse={best_sample_metrics['mse']:.4f}"
        )

        if trial_score > best_score:
            best_score = trial_score
            best_cfg = dict(cfg)
            best_batch = int(trial_batch)
            best_visual_payload = {
                "best_sample": display_sample,
                "metrics": display_metrics,
                "loss_curve": [float(x) for x in losses],
            }

    if best_cfg is None or best_visual_payload is None:
        raise RuntimeError("Diffusion tuning failed to produce a valid trial.")

    best_model, best_betas, best_alphas, best_alpha_bars, _, _ = _train_diffusion_with_fallback(best_cfg, best_batch)

    future_windows = sample_ddpm(
        model=best_model,
        sample_count=max(24, future_samples),
        height=window,
        width=49,
        betas=best_betas,
        alphas=best_alphas,
        alpha_bars=best_alpha_bars,
    )
    pred_win, pred_addl, next_win_prob, next_row_prior = diffusion_predict_next_day(
        samples=future_windows,
        actual_last_row=target_role[-1],
        recent_binary=target_binary,
        sample_limit=future_samples,
    )

    row_idx = np.arange(len(df) - window, len(df))
    row_labels = [str(int(df.loc[i, "Draw"])) if not pd.isna(df.loc[i, "Draw"]) else str(i) for i in row_idx]
    trial_df = pd.DataFrame(trial_rows).sort_values(["best_score", "consensus_score", "best_hit_rate"], ascending=[False, False, False]).reset_index(drop=True)
    best_trial_row = trial_df.iloc[0].to_dict() if not trial_df.empty else {}

    return {
        "window": window,
        "row_labels": row_labels,
        "target_role": target_role,
        "generated_role": np.asarray(best_visual_payload["best_sample"], dtype=np.float32),
        "accuracy_map": np.asarray(best_visual_payload["metrics"]["accuracy_map"], dtype=np.float32),
        "overlap_map": np.asarray(best_visual_payload["metrics"]["overlap_map"], dtype=np.float32),
        "row_hits": np.asarray(best_visual_payload["metrics"]["row_hits"], dtype=np.float32),
        "trial_df": trial_df,
        "trial_best": best_trial_row,
        "trial_loss_curve": best_visual_payload["loss_curve"],
        "trial_hit_rate": float(best_visual_payload["metrics"]["hit_rate"]),
        "trial_mse": float(best_visual_payload["metrics"]["mse"]),
        "trial_score": float(best_visual_payload["metrics"]["score"]),
        "diffusion_pred_win": pred_win,
        "diffusion_pred_addl": pred_addl,
        "diffusion_next_win_prob": next_win_prob,
        "diffusion_next_row_prior": next_row_prior,
        "future_windows_used": int(max(24, future_samples)),
    }


def feature_group_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "primary": float(args.w_primary),
        "other": float(args.w_other),
        "repel": float(args.w_repel),
        "cluster": float(args.w_cluster),
        "line": float(args.w_line),
    }


def generate_feature_weight_candidates(base: Dict[str, float], trials: int) -> List[Dict[str, float]]:
    cands: List[Dict[str, float]] = [dict(base)]
    rng = np.random.default_rng(ACTIVE_SEED + 119)
    keys = list(base.keys())
    for _ in range(max(0, trials - 1)):
        cand: Dict[str, float] = {}
        for k in keys:
            v = float(base[k])
            scale = float(np.exp(rng.normal(0.0, 0.33)))
            cand[k] = float(np.clip(v * scale, 0.55, 6.50))
        cands.append(cand)
    return cands


def build_feature_pipeline(
    df: pd.DataFrame,
    primary_cols: List[str],
    other_cols: List[str],
    group_weights: Dict[str, float],
    forced_clusters: int,
) -> Dict[str, object]:
    base_feature_cols = primary_cols + other_cols
    X_base = df[base_feature_cols].values.astype(np.float32)
    base_feature_weights = np.array(
        [group_weights["primary"]] * len(primary_cols) + [group_weights["other"]] * len(other_cols),
        dtype=np.float32,
    )
    X_base_weighted = X_base * base_feature_weights

    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_base_weighted)
    cluster_model, best_k, silhouette = build_cluster_model(X_cluster_scaled, forced_clusters)
    cluster_labels = cluster_model.labels_

    line_system = build_line_endpoint_system(df)
    line_prior_matrix = line_system["line_prior"]
    line_merge_matrix = line_system["line_merge"]
    cooc_system = build_cooccurrence_system(df)
    cooc_prior_matrix = cooc_system["cooc_prior"]
    repel_system = build_repel_system(df)
    repel_prior_matrix = repel_system["repel_prior"]

    win_cluster_prior_matrix, addl_cluster_prior_matrix, cluster_transition_matrix = build_dynamic_cluster_priors(
        df=df,
        cluster_labels=cluster_labels,
        k=best_k,
    )

    line_cols = [f"LinePrior_{i+1}" for i in range(49)]
    line_merge_cols = [f"LineMerge_{i+1}" for i in range(49)]
    cooc_cols = [f"Cooc_{i+1}" for i in range(49)]
    repel_cols = [f"Repel_{i+1}" for i in range(49)]
    cluster_prior_cols = [f"ClusterPrior_{i+1}" for i in range(49)]

    extra_features = pd.DataFrame(
        np.hstack(
            [
                line_prior_matrix,
                line_merge_matrix,
                cooc_prior_matrix,
                repel_prior_matrix,
                win_cluster_prior_matrix,
            ]
        ),
        columns=line_cols + line_merge_cols + cooc_cols + repel_cols + cluster_prior_cols,
        index=df.index,
    )
    df_features = pd.concat([df.copy(), extra_features], axis=1)

    feature_cols = (
        base_feature_cols
        + line_cols
        + line_merge_cols
        + cooc_cols
        + repel_cols
        + cluster_prior_cols
    )
    feature_weights = np.array(
        [group_weights["primary"]] * len(primary_cols)
        + [group_weights["other"]] * len(other_cols)
        + [group_weights["line"]] * len(line_cols)
        + [group_weights["line"]] * len(line_merge_cols)
        + [group_weights["line"]] * len(cooc_cols)
        + [group_weights["repel"]] * len(repel_cols)
        + [group_weights["cluster"]] * len(cluster_prior_cols),
        dtype=np.float32,
    )
    X = df_features[feature_cols].values.astype(np.float32)
    X_weighted = X * feature_weights

    pca = PCA(n_components=2, random_state=ACTIVE_SEED)
    embedding = pca.fit_transform(X_cluster_scaled)
    cluster_profile_means = (
        df_features[PRIMARY_COLS]
        .assign(Cluster=cluster_labels)
        .groupby("Cluster")[PRIMARY_COLS]
        .mean()
        .sort_index()
    )

    return {
        "df_features": df_features,
        "feature_cols": feature_cols,
        "X_weighted": X_weighted,
        "cluster_labels": cluster_labels,
        "best_k": int(best_k),
        "silhouette": float(silhouette),
        "embedding": embedding,
        "cluster_profile_means": cluster_profile_means,
        "line_prior_matrix": line_prior_matrix,
        "line_merge_matrix": line_merge_matrix,
        "repel_prior_matrix": repel_prior_matrix,
        "win_cluster_prior_matrix": win_cluster_prior_matrix,
        "addl_cluster_prior_matrix": addl_cluster_prior_matrix,
        "cluster_transition_matrix": cluster_transition_matrix,
    }


def tune_feature_group_weights(
    df: pd.DataFrame,
    primary_cols: List[str],
    other_cols: List[str],
    forced_clusters: int,
    base_weights: Dict[str, float],
    trials: int,
    epochs: int,
    focus_last_n: int,
    batch_size: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if trials <= 0:
        return dict(base_weights), pd.DataFrame()

    candidates = generate_feature_weight_candidates(base_weights, max(1, trials))
    proxy_cfg = ModelConfig(seq_len=24, lstm_units=96, gru_units=64, dense_units=160, dropout=0.28, lr=3e-4)
    rows: List[Dict[str, object]] = []

    best_weights = dict(base_weights)
    best_score = -1e18

    for trial_i, cand in enumerate(candidates, start=1):
        try:
            pipe = build_feature_pipeline(
                df=df,
                primary_cols=primary_cols,
                other_cols=other_cols,
                group_weights=cand,
                forced_clusters=forced_clusters,
            )
            X_seq, y_raw, target_rows = build_sequences(pipe["X_weighted"], pipe["df_features"], proxy_cfg.seq_len, other_cols)
            if len(X_seq) < 200:
                continue
            train_idx, val_idx, _ = split_train_val_test(len(X_seq))
            model, history, _, scaler_x = run_single_fit(
                config=proxy_cfg,
                X_seq=X_seq,
                y_raw=y_raw,
                train_idx=train_idx,
                val_idx=val_idx,
                epochs=max(2, int(epochs)),
                batch_size=batch_size,
                verbose=0,
            )
            eval_idx = recent_seq_indices(
                len(X_seq),
                max(12, focus_last_n if focus_last_n > 0 else 20),
                min_train=max(120, proxy_cfg.seq_len * 4),
            )
            if len(eval_idx) == 0:
                eval_idx = val_idx
            hit_metrics = evaluate_hit_score(
                model=model,
                scaler_x=scaler_x,
                X_seq=X_seq,
                eval_idx=eval_idx,
                target_rows=target_rows,
                df=pipe["df_features"],
                win_cluster_prior_matrix=pipe["win_cluster_prior_matrix"],
                addl_cluster_prior_matrix=pipe["addl_cluster_prior_matrix"],
                repel_prior_matrix=pipe["repel_prior_matrix"],
                weights=BLEND_WEIGHTS,
            )
            val_loss = float(np.min(history.history.get("val_loss", [math.inf])))
            score = (
                120.0 * hit_metrics["p_hit_ge3"]
                + 230.0 * hit_metrics["p_hit_ge4"]
                + 1.4 * hit_metrics["avg_win_hits"]
                + 2.0 * hit_metrics["addl_acc"]
                + 0.25 * float(pipe["silhouette"])
                - 0.015 * val_loss
            )
            row = {
                "trial": trial_i,
                "score": float(score),
                "proxy_val_loss": val_loss,
                "proxy_avg_hits": float(hit_metrics["avg_win_hits"]),
                "proxy_p_hit_ge3": float(hit_metrics["p_hit_ge3"]),
                "proxy_p_hit_ge4": float(hit_metrics["p_hit_ge4"]),
                "proxy_addl_acc": float(hit_metrics["addl_acc"]),
                "silhouette": float(pipe["silhouette"]),
            }
            for k in DEFAULT_FEATURE_GROUP_WEIGHTS.keys():
                row[f"w_{k}"] = float(cand[k])
            rows.append(row)
            print(
                f"[WEIGHT-TUNE] Trial {trial_i}/{len(candidates)} "
                f"score={score:.4f} p>=3={hit_metrics['p_hit_ge3']:.4f} "
                f"p>=4={hit_metrics['p_hit_ge4']:.4f} avg={hit_metrics['avg_win_hits']:.4f}"
            )
            if score > best_score:
                best_score = float(score)
                best_weights = dict(cand)
        except Exception as ex:
            print(f"[WEIGHT-TUNE] Trial {trial_i}/{len(candidates)} failed: {ex}")

    tune_df = pd.DataFrame(rows)
    if not tune_df.empty:
        tune_df = tune_df.sort_values(["score", "proxy_p_hit_ge4", "proxy_p_hit_ge3", "proxy_avg_hits"], ascending=[False, False, False, False]).reset_index(drop=True)
    return best_weights, tune_df


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_cluster_model(X_scaled: np.ndarray, forced_clusters: int = 0) -> Tuple[KMeans, int, float]:
    if forced_clusters > 1:
        model = KMeans(n_clusters=forced_clusters, random_state=SEED, n_init=30)
        labels = model.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        return model, forced_clusters, sil

    best_model: KMeans | None = None
    best_k = 4
    best_sil = -1.0
    for k in range(4, 11):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=30)
        labels = model.fit_predict(X_scaled)
        score = float(silhouette_score(X_scaled, labels))
        if score > best_sil:
            best_model = model
            best_k = k
            best_sil = score
    if best_model is None:
        raise RuntimeError("Unable to fit cluster model.")
    return best_model, best_k, best_sil


def build_repel_system(
    df: pd.DataFrame,
    repel_window: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Build recency-frequency repel prior using only rows before each target row.
    Higher recent frequency -> stronger repel signal for that number.
    """
    n = len(df)
    draw_win = df[WIN_COLS].values.astype(np.int32)
    repel = np.zeros((n, 49), dtype=np.float32)
    uniform = np.ones(49, dtype=np.float32) / 49.0

    for t in range(n):
        if t <= 0:
            repel[t] = uniform
            continue
        start = max(0, t - repel_window)
        recent = draw_win[start:t].reshape(-1)
        if len(recent) == 0:
            repel[t] = uniform
            continue
        counts = np.bincount(recent, minlength=50)[1:].astype(np.float32)
        s = float(np.sum(counts))
        if s <= 0.0:
            repel[t] = uniform
        else:
            repel[t] = counts / s
    return {"repel_prior": repel.astype(np.float32)}


def build_line_endpoint_system(
    df: pd.DataFrame,
    lookback: int = 26,
    recency_decay: float = 0.88,
    sigma: float = 1.08,
    slope_clip: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Endpoint-convergence prior inspired by manual line-drawing strategy.
    For each step t, project lines from historical consecutive draw pairs:
      a -> b  (from t-lag-1 to t-lag), endpoint candidate = b + (b-a)
    Numbers receiving many projected endpoints become high-probability targets.
    """
    n = len(df)
    draw_all = df[WIN_COLS + ["Addl No."]].values.astype(np.int32)
    num_axis = np.arange(1, 50, dtype=np.float32)

    endpoint_field = np.zeros((n, 49), dtype=np.float32)
    merge_field = np.zeros((n, 49), dtype=np.float32)
    uniform = np.ones(49, dtype=np.float32) / 49.0

    for t in range(n):
        if t < 2:
            endpoint_field[t] = uniform
            merge_field[t] = uniform
            continue

        smooth_score = np.zeros(49, dtype=np.float64)
        merge_count = np.zeros(49, dtype=np.float64)

        max_lag = min(lookback, t - 1)
        for lag in range(1, max_lag + 1):
            w = recency_decay ** (lag - 1)
            row_a = draw_all[t - lag - 1]
            row_b = draw_all[t - lag]
            for a in row_a:
                for b in row_b:
                    delta = int(b) - int(a)
                    if abs(delta) > slope_clip:
                        continue
                    endpoint = int(b) + delta
                    if endpoint < 1 or endpoint > 49:
                        continue
                    dist = np.abs(num_axis - float(endpoint))
                    smooth_score += w * np.exp(-(dist * dist) / (2.0 * sigma * sigma))
                    merge_count[endpoint - 1] += w

        if smooth_score.sum() > 1e-12:
            endpoint_field[t] = (smooth_score / smooth_score.sum()).astype(np.float32)
        else:
            endpoint_field[t] = uniform

        if merge_count.sum() > 1e-12:
            merge_field[t] = (merge_count / merge_count.sum()).astype(np.float32)
        else:
            merge_field[t] = uniform

    line_prior = 0.74 * endpoint_field + 0.26 * merge_field
    line_prior = line_prior / (line_prior.sum(axis=1, keepdims=True) + 1e-8)
    return {
        "line_prior": line_prior.astype(np.float32),
        "line_endpoint": endpoint_field.astype(np.float32),
        "line_merge": merge_field.astype(np.float32),
    }


def build_cooccurrence_system(
    df: pd.DataFrame,
    decay: float = 0.989,
    addl_strength: float = 0.30,
    pair_strength: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Build a recency-aware co-occurrence prior over numbers 1..49.
    For row t, the prior uses only rows [0..t-1], with exponential decay.
    """
    n = len(df)
    draw_win = df[WIN_COLS].values.astype(np.int32) - 1
    draw_addl = df["Addl No."].values.astype(np.int32) - 1

    cooc_prior = np.zeros((n, 49), dtype=np.float32)
    freq = np.ones(49, dtype=np.float64)
    pair = np.ones((49, 49), dtype=np.float64)
    uniform = np.ones(49, dtype=np.float64) / 49.0

    for t in range(n):
        if t == 0:
            cooc_prior[t] = uniform.astype(np.float32)
        else:
            base = freq / (freq.sum() + 1e-12)
            pair_row = pair / (pair.sum(axis=1, keepdims=True) + 1e-12)
            # Blend graph-propagated tendency with direct frequency baseline.
            score = 0.72 * (pair_row @ base) + 0.28 * base
            score = score / (score.sum() + 1e-12)
            cooc_prior[t] = score.astype(np.float32)

        # Decay old evidence before consuming current row.
        freq *= decay
        pair *= decay

        wins = np.unique(draw_win[t]).astype(np.int32)
        addl = int(draw_addl[t])

        freq[wins] += 1.0
        freq[addl] += addl_strength

        for a in wins:
            for b in wins:
                if a != b:
                    pair[a, b] += pair_strength
            pair[a, addl] += 0.42 * pair_strength
            pair[addl, a] += 0.26 * pair_strength

    return {"cooc_prior": cooc_prior.astype(np.float32)}


def build_dynamic_cluster_priors(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-row cluster priors using only rows before each time step.
    Returns:
      win_cluster_prior: (n,49)
      addl_cluster_prior: (n,49)
      transition_matrix: final normalized cluster transition matrix
    """
    n = len(df)
    win_prior = np.zeros((n, 49), dtype=np.float32)
    addl_prior = np.zeros((n, 49), dtype=np.float32)

    trans = np.ones((k, k), dtype=np.float64)
    win_counts = np.ones((k, 49), dtype=np.float64)
    addl_counts = np.ones((k, 49), dtype=np.float64)

    for t in range(n):
        if t == 0:
            win_prior[t] = np.ones(49, dtype=np.float32) / 49.0
            addl_prior[t] = np.ones(49, dtype=np.float32) / 49.0
        else:
            last_cluster = int(cluster_labels[t - 1])
            next_cluster = int(np.argmax(trans[last_cluster]))
            w = win_counts[next_cluster]
            a = addl_counts[next_cluster]
            win_prior[t] = (w / w.sum()).astype(np.float32)
            addl_prior[t] = (a / a.sum()).astype(np.float32)

        c = int(cluster_labels[t])
        wins = df.loc[t, WIN_COLS].astype(int).values - 1
        addl = int(df.loc[t, "Addl No."]) - 1
        win_counts[c, wins] += 1.0
        addl_counts[c, addl] += 1.0

        if t > 0:
            prev_c = int(cluster_labels[t - 1])
            trans[prev_c, c] += 1.0

    trans = trans / trans.sum(axis=1, keepdims=True)
    return win_prior, addl_prior, trans.astype(np.float32)


def cluster_scatter_img(embedding: np.ndarray, labels: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=22, alpha=0.86, edgecolor="none")
    ax.set_title("Cluster Shape Map (PCA Projection)", fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cluster")
    fig.tight_layout()
    return fig_to_base64(fig)


def cluster_timeline_img(labels: np.ndarray) -> str:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, labels, linewidth=0.9, color="#243b53", alpha=0.75)
    ax.scatter(x, labels, c=labels, cmap="tab20", s=9, alpha=0.9)
    ax.set_title("Cluster Trend Across Draw Timeline (Oldest -> Newest)", fontweight="bold")
    ax.set_xlabel("Draw Sequence Index")
    ax.set_ylabel("Cluster ID")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    return fig_to_base64(fig)


def cluster_profile_img(df: pd.DataFrame, labels: np.ndarray) -> str:
    means = df[PRIMARY_COLS].assign(Cluster=labels).groupby("Cluster")[PRIMARY_COLS].mean().sort_index()
    values = means.values
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(PRIMARY_COLS)))
    ax.set_xticklabels(PRIMARY_COLS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(means.index)))
    ax.set_yticklabels([str(v) for v in means.index])
    ax.set_title("Cluster Pattern Focus (Win_1..Win_6 + Addl No.)", fontweight="bold")
    ax.set_xlabel("Key Number Columns")
    ax.set_ylabel("Cluster")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:.1f}", ha="center", va="center", fontsize=8, color="#0b132b")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mean")
    fig.tight_layout()
    return fig_to_base64(fig)


def training_curve_img(history: keras.callbacks.History) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history.history.get("loss", []), label="train_loss", linewidth=1.6)
    ax.plot(history.history.get("val_loss", []), label="val_loss", linewidth=1.6)
    ax.set_title("Final Training Curve", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)


def tuning_curve_img(tuning_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(tuning_df))
    ax.plot(x, tuning_df["mean_val_hit_score"], marker="o", linewidth=1.2, color="#136f63")
    for i, row in tuning_df.iterrows():
        ax.text(i, row["mean_val_hit_score"], f"seq={int(row['seq_len'])}", fontsize=7, alpha=0.8)
    ax.set_title("Hyperparameter Tuning Result (Higher Hit Score is Better)", fontweight="bold")
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Mean Validation Hit Score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig_to_base64(fig)


def backtest_plot_img(backtest_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(backtest_df["win_hits"], bins=np.arange(-0.5, 7.5, 1), color="#2f6f6f", alpha=0.88, edgecolor="#f8f9fa")
    axes[0].set_title("Backtest Win-Hit Distribution")
    axes[0].set_xlabel("Matched Winning Numbers (0-6)")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)

    axes[1].plot(backtest_df.index.values, backtest_df["win_hits"], color="#1f4e79", linewidth=1.0, alpha=0.85)
    rolling = backtest_df["win_hits"].rolling(20, min_periods=1).mean()
    axes[1].plot(backtest_df.index.values, rolling, color="#d1495b", linewidth=1.8, label="rolling mean (20)")
    axes[1].set_title("Backtest Hit Trend")
    axes[1].set_xlabel("Backtest Sample Index")
    axes[1].set_ylabel("Win Hits")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    return fig_to_base64(fig)


def build_sequences(X_weighted: np.ndarray, df: pd.DataFrame, seq_len: int, aux_cols: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    X_seq: List[np.ndarray] = []
    y: Dict[str, List[np.ndarray]] = {f"win_{i}": [] for i in range(1, 7)}
    y["addl"] = []
    y["win_set"] = []
    y["aux"] = []
    target_rows: List[int] = []

    for target_idx in range(seq_len, len(df)):
        X_seq.append(X_weighted[target_idx - seq_len : target_idx])
        win_set = np.zeros(49, dtype=np.float32)
        for i, col in enumerate(WIN_COLS, start=1):
            win_value = int(df.at[target_idx, col]) - 1
            y[f"win_{i}"].append(win_value)
            win_set[win_value] = 1.0
        y["addl"].append(int(df.at[target_idx, "Addl No."]) - 1)
        y["win_set"].append(win_set)
        y["aux"].append(df.loc[target_idx, aux_cols].values.astype(np.float32))
        target_rows.append(target_idx)

    X_seq_arr = np.asarray(X_seq, dtype=np.float32)
    y_arr: Dict[str, np.ndarray] = {}
    for key, vals in y.items():
        if key in {"aux", "win_set"}:
            y_arr[key] = np.asarray(vals, dtype=np.float32)
        else:
            y_arr[key] = np.asarray(vals, dtype=np.int32)
    return X_seq_arr, y_arr, np.asarray(target_rows, dtype=np.int32)


def make_walk_forward_splits(n_samples: int, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    val_size = max(24, int(n_samples * 0.08))
    start = max(val_size * 2, int(n_samples * 0.58))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        train_end = start + i * val_size
        val_start = train_end
        val_end = min(n_samples, val_start + val_size)
        if train_end < 120 or (val_end - val_start) < 16:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
        if val_end >= n_samples:
            break
    return splits


def split_train_val_test(n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(1, int(n_samples * 0.8))
    val_end = min(n_samples - 1, max(train_end + 1, int(n_samples * 0.9)))
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)
    return train_idx, val_idx, test_idx


def recent_seq_indices(n_samples: int, last_n: int, min_train: int = 120) -> np.ndarray:
    if n_samples <= 0 or last_n <= 0:
        return np.array([], dtype=np.int32)
    start = max(min_train, n_samples - int(last_n))
    if start >= n_samples:
        return np.array([], dtype=np.int32)
    return np.arange(start, n_samples, dtype=np.int32)


def one_step_train_val_indices(
    seq_index: int,
    min_train_core: int = 120,
    min_val: int = 16,
    max_val: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    train_end = int(seq_index)
    if train_end <= min_train_core + min_val:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    val_size = max(min_val, min(max_val, int(train_end * 0.12)))
    train_core_end = train_end - val_size
    if train_core_end < min_train_core:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    train_idx = np.arange(0, train_core_end, dtype=np.int32)
    val_idx = np.arange(train_core_end, train_end, dtype=np.int32)
    return train_idx, val_idx


def build_model(config: ModelConfig, n_features: int, n_aux: int, steps_per_execution: int = 0) -> keras.Model:
    """
    Hybrid architecture inspired by:
    - PatchTST (patch tokenization + transformer mixing),
    - Informer/Patch-style temporal attention,
    - TCN dilated-convolution memory branch.
    """
    key_dim = max(16, config.lstm_units // 8)

    inp = keras.Input(shape=(config.seq_len, n_features), name="sequence")

    # Branch A: recurrent temporal memory (local order detail)
    r = layers.Bidirectional(
        layers.LSTM(config.lstm_units, return_sequences=True, dropout=config.dropout),
        name="bilstm_1",
    )(inp)
    r_attn = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=key_dim,
        dropout=min(0.3, config.dropout),
        name="self_attn_r",
    )(r, r)
    r = layers.Add(name="r_residual")([r, r_attn])
    r = layers.LayerNormalization(name="r_norm")(r)
    r = layers.Bidirectional(
        layers.GRU(config.gru_units, return_sequences=False, dropout=config.dropout),
        name="bigru_2",
    )(r)

    # Branch B: PatchTST-like token branch
    p = layers.Conv1D(
        filters=max(96, config.dense_units),
        kernel_size=4,
        strides=2,
        padding="causal",
        activation="gelu",
        name="patch_embed",
    )(inp)
    p = layers.LayerNormalization(name="p_norm_0")(p)
    for block_idx in range(2):
        p_attn = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=max(16, config.dense_units // 8),
            dropout=min(0.3, config.dropout),
            name=f"p_attn_{block_idx}",
        )(p, p)
        p = layers.Add(name=f"p_res_{block_idx}_a")([p, p_attn])
        p = layers.LayerNormalization(name=f"p_norm_{block_idx}_a")(p)
        p_ff = layers.Dense(max(128, config.dense_units * 2), activation="gelu", name=f"p_ff_{block_idx}_1")(p)
        p_ff = layers.Dropout(config.dropout, name=f"p_ff_drop_{block_idx}")(p_ff)
        p_ff = layers.Dense(max(96, config.dense_units), activation="linear", name=f"p_ff_{block_idx}_2")(p_ff)
        p = layers.Add(name=f"p_res_{block_idx}_b")([p, p_ff])
        p = layers.LayerNormalization(name=f"p_norm_{block_idx}_b")(p)
    p = layers.GlobalAveragePooling1D(name="p_pool")(p)

    # Branch C: TCN-like dilated causal convolutions
    t = layers.Conv1D(max(64, config.gru_units), 3, padding="causal", dilation_rate=1, activation="gelu", name="tcn_d1")(inp)
    t = layers.Conv1D(max(64, config.gru_units), 3, padding="causal", dilation_rate=2, activation="gelu", name="tcn_d2")(t)
    t = layers.Conv1D(max(64, config.gru_units), 3, padding="causal", dilation_rate=4, activation="gelu", name="tcn_d4")(t)
    t = layers.GlobalMaxPooling1D(name="tcn_pool")(t)

    # Branch D: explicit temporal focus-attention. This gives a learnable "where to look"
    # weighting over timesteps before fusion.
    a = layers.LayerNormalization(name="focus_norm_0")(inp)
    a = layers.Dense(max(64, config.gru_units), activation="gelu", name="focus_proj")(a)
    a_logits = layers.Dense(1, activation="linear", name="focus_logits")(a)
    a_weights = layers.Softmax(axis=1, name="focus_weights")(a_logits)
    a_ctx = layers.Multiply(name="focus_weighted")([a, a_weights])
    a_ctx = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1), name="focus_context")(a_ctx)

    x = layers.Concatenate(name="fusion_concat")([r, p, t, a_ctx])
    x = layers.Dense(config.dense_units, activation="gelu", name="dense_1")(x)
    x = layers.Dropout(min(0.45, config.dropout + 0.08), name="drop_1")(x)
    x = layers.Dense(max(96, config.dense_units - 32), activation="gelu", name="dense_2")(x)
    trunk = layers.Dropout(config.dropout, name="drop_2")(x)

    outputs: Dict[str, layers.Layer] = {}
    for i in range(1, 7):
        outputs[f"win_{i}"] = layers.Dense(49, activation="softmax", dtype="float32", name=f"win_{i}")(trunk)
    outputs["win_set"] = layers.Dense(49, activation="sigmoid", dtype="float32", name="win_set")(trunk)
    outputs["addl"] = layers.Dense(49, activation="softmax", dtype="float32", name="addl")(trunk)
    outputs["aux"] = layers.Dense(n_aux, activation="linear", dtype="float32", name="aux")(trunk)

    model = keras.Model(inputs=inp, outputs=outputs, name="ToTo_Tuned_Cluster_DL")

    losses = {f"win_{i}": "sparse_categorical_crossentropy" for i in range(1, 7)}
    losses["win_set"] = "binary_crossentropy"
    losses["addl"] = "sparse_categorical_crossentropy"
    losses["aux"] = "mse"

    loss_weights = {f"win_{i}": 1.9 for i in range(1, 7)}
    loss_weights["win_set"] = 3.2
    loss_weights["addl"] = 2.4
    loss_weights["aux"] = 0.28

    metrics = {f"win_{i}": [keras.metrics.SparseCategoricalAccuracy(name="acc")] for i in range(1, 7)}
    metrics["win_set"] = [keras.metrics.BinaryAccuracy(name="acc")]
    metrics["addl"] = [keras.metrics.SparseCategoricalAccuracy(name="acc")]
    metrics["aux"] = [keras.metrics.MeanAbsoluteError(name="mae")]

    compile_kwargs = dict(
        optimizer=keras.optimizers.Adam(learning_rate=config.lr, clipnorm=1.0),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
        weighted_metrics=[],
    )
    if int(steps_per_execution) > 0:
        compile_kwargs["steps_per_execution"] = int(steps_per_execution)
    model.compile(**compile_kwargs)
    return model


def make_dataset(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    cache: bool = False,
    sample_weights: np.ndarray | None = None,
) -> tf.data.Dataset:
    if sample_weights is None:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        w = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if len(w) != len(X):
            w = np.ones(len(X), dtype=np.float32)
        w_dict = {k: w for k in y.keys()}
        ds = tf.data.Dataset.from_tensor_slices((X, y, w_dict))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(len(X), seed=ACTIVE_SEED, reshuffle_each_iteration=True)
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_outputs_dict(model: keras.Model, x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fast inference path that avoids repeated Keras predict-function retracing.
    """
    out = model(x, training=False)
    if isinstance(out, dict):
        return {k: np.asarray(v) for k, v in out.items()}
    if isinstance(out, (list, tuple)):
        names = list(model.output_names)
        return {names[i]: np.asarray(v) for i, v in enumerate(out)}
    names = list(model.output_names)
    key = names[0] if names else "output_0"
    return {key: np.asarray(out)}


def scale_fold_data(
    X_seq: np.ndarray,
    y_raw: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    other_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], StandardScaler]:
    n_features = X_seq.shape[-1]
    scaler_x = StandardScaler()
    scaler_x.fit(X_seq[train_idx].reshape(-1, n_features))

    X_train = scaler_x.transform(X_seq[train_idx].reshape(-1, n_features)).reshape(len(train_idx), X_seq.shape[1], n_features).astype(np.float32)
    X_other = scaler_x.transform(X_seq[other_idx].reshape(-1, n_features)).reshape(len(other_idx), X_seq.shape[1], n_features).astype(np.float32)

    aux_scaler = StandardScaler()
    aux_scaler.fit(y_raw["aux"][train_idx])

    y_train: Dict[str, np.ndarray] = {}
    y_other: Dict[str, np.ndarray] = {}
    for key, arr in y_raw.items():
        if key == "aux":
            y_train[key] = aux_scaler.transform(arr[train_idx]).astype(np.float32)
            y_other[key] = aux_scaler.transform(arr[other_idx]).astype(np.float32)
        elif key == "win_set":
            y_train[key] = arr[train_idx].astype(np.float32)
            y_other[key] = arr[other_idx].astype(np.float32)
        else:
            y_train[key] = arr[train_idx].astype(np.int32)
            y_other[key] = arr[other_idx].astype(np.int32)
    return X_train, y_train, X_other, y_other, scaler_x


def run_single_fit(
    config: ModelConfig,
    X_seq: np.ndarray,
    y_raw: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    steps_per_execution: int = 0,
    cache_dataset: bool = False,
    train_sample_weights: np.ndarray | None = None,
    val_sample_weights: np.ndarray | None = None,
    verbose: int = 0,
) -> Tuple[keras.Model, keras.callbacks.History, Dict[str, float], StandardScaler]:
    X_train, y_train, X_val, y_val, scaler_x = scale_fold_data(X_seq, y_raw, train_idx, val_idx)
    cur_batch = int(max(8, batch_size))
    last_exc: Exception | None = None

    while cur_batch >= 8:
        tf.keras.backend.clear_session()
        model = build_model(
            config,
            n_features=X_seq.shape[-1],
            n_aux=y_raw["aux"].shape[-1],
            steps_per_execution=steps_per_execution,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        ]
        train_ds = make_dataset(
            X_train,
            y_train,
            batch_size=cur_batch,
            shuffle=True,
            cache=cache_dataset,
            sample_weights=train_sample_weights,
        )
        val_ds = make_dataset(
            X_val,
            y_val,
            batch_size=cur_batch,
            shuffle=False,
            cache=cache_dataset,
            sample_weights=val_sample_weights,
        )
        try:
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=verbose)
            val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
            return model, history, val_metrics, scaler_x
        except tf.errors.ResourceExhaustedError as exc:
            last_exc = exc
        except Exception as exc:
            msg = str(exc).lower()
            if "resourceexhausted" not in msg and "oom" not in msg and "out of memory" not in msg:
                raise
            last_exc = exc

        next_batch = int(max(8, cur_batch // 2))
        if next_batch == cur_batch:
            break
        print(f"[GPU] OOM at batch_size={cur_batch}; retrying with batch_size={next_batch}")
        cur_batch = next_batch

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("run_single_fit failed before model training started.")


def transition_from_labels(labels: np.ndarray, k: int) -> Tuple[np.ndarray, int, int]:
    trans = np.ones((k, k), dtype=np.float64)
    for a, b in zip(labels[:-1], labels[1:]):
        trans[a, b] += 1.0
    trans /= trans.sum(axis=1, keepdims=True)
    last_cluster = int(labels[-1])
    next_cluster = int(np.argmax(trans[last_cluster]))
    return trans, last_cluster, next_cluster


def priors_from_history(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    next_cluster: int,
    cutoff_row: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # cutoff_row is exclusive (use rows [0, cutoff_row))
    if cutoff_row <= 0:
        uniform = np.ones(49, dtype=np.float32) / 49.0
        return uniform, uniform

    hist_clusters = cluster_labels[:cutoff_row]
    mask = hist_clusters == next_cluster
    if not np.any(mask):
        uniform = np.ones(49, dtype=np.float32) / 49.0
        return uniform, uniform

    hist_df = df.iloc[:cutoff_row]
    wins = hist_df.loc[mask, WIN_COLS].values.astype(int).reshape(-1)
    addl = hist_df.loc[mask, "Addl No."].values.astype(int)

    win_counts = np.bincount(wins, minlength=50)[1:].astype(np.float32)
    addl_counts = np.bincount(addl, minlength=50)[1:].astype(np.float32)
    win_prior = win_counts / win_counts.sum() if win_counts.sum() > 0 else np.ones(49, dtype=np.float32) / 49.0
    addl_prior = addl_counts / addl_counts.sum() if addl_counts.sum() > 0 else np.ones(49, dtype=np.float32) / 49.0
    return win_prior, addl_prior


def normalize_prob(v: np.ndarray) -> np.ndarray:
    v = np.clip(v.astype(np.float64), 1e-12, None)
    s = float(v.sum())
    if s <= 0:
        return np.ones_like(v, dtype=np.float64) / float(len(v))
    return (v / s).astype(np.float64)


def select_win_numbers_with_repulsion(
    scores: np.ndarray,
    count: int = 6,
    p1: float = 0.88,
    p2: float = 0.94,
    p3: float = 0.97,
) -> List[int]:
    """
    Greedy selection with short-range repulsion to avoid collapsing into one narrow band.
    """
    work = scores.astype(np.float64).copy()
    selected: List[int] = []
    for _ in range(count):
        idx = int(np.argmax(work))
        num = idx + 1
        selected.append(num)
        work[idx] = -1.0
        for j in range(49):
            dist = abs((j + 1) - num)
            if dist == 0:
                continue
            if dist <= 1:
                work[j] *= float(np.clip(p1, 0.75, 0.999))
            elif dist <= 2:
                work[j] *= float(np.clip(p2, 0.80, 0.999))
            elif dist <= 3:
                work[j] *= float(np.clip(p3, 0.85, 0.999))
    return sorted(selected)


def combine_prediction(
    pred: Dict[str, np.ndarray],
    win_cluster_prior: np.ndarray,
    addl_cluster_prior: np.ndarray,
    repel_prior: np.ndarray,
    weights: Dict[str, float],
) -> Tuple[List[int], int]:
    win_soft = np.mean([pred[f"win_{i}"][0] for i in range(1, 7)], axis=0)
    win_set = pred["win_set"][0]
    win_set_prob = normalize_prob(win_set)
    model_conf = float(np.max(win_set_prob))
    # High confidence => reduce repulsion and sharpen; low confidence => keep wider spread.
    repel_scale = 0.46 + 0.84 * (1.0 - model_conf)

    # Pruned blend: model + cluster attraction minus repulsion.
    combined_win = (
        weights["model_soft"] * normalize_prob(win_soft)
        + weights["model_set"] * win_set_prob
        + weights["cluster"] * normalize_prob(win_cluster_prior)
        - (weights["repel"] * repel_scale) * normalize_prob(repel_prior)
    )
    combined_win = normalize_prob(np.clip(combined_win, 1e-12, None))
    sharpen_t = float(np.clip(0.84 + 0.34 * (1.0 - model_conf), 0.70, 1.20))
    combined_win = np.power(np.clip(combined_win, 1e-12, None), 1.0 / sharpen_t)
    combined_win = normalize_prob(combined_win)
    cluster_prob = normalize_prob(win_cluster_prior)

    plain_top = sorted([int(i + 1) for i in np.argsort(combined_win)[-6:].tolist()])
    repel_light = select_win_numbers_with_repulsion(combined_win, count=6, p1=0.93, p2=0.97, p3=0.985)
    repel_std = select_win_numbers_with_repulsion(combined_win, count=6, p1=0.88, p2=0.94, p3=0.97)

    def candidate_utility(nums: List[int]) -> float:
        idx = np.asarray(nums, dtype=np.int32) - 1
        p = np.clip(combined_win[idx], 1e-12, 1.0)
        c = np.clip(cluster_prob[idx], 1e-12, 1.0)
        # Tail-hit mode prefers concentrated high-probability sets.
        util = (
            2.7 * float(np.mean(np.log(p)))
            + 0.32 * float(np.mean(np.log(c)))
        )
        return util

    candidates: Dict[Tuple[int, ...], float] = {}
    for cand in (plain_top, repel_light, repel_std):
        key = tuple(sorted(cand))
        candidates[key] = candidate_utility(list(key))
    if model_conf >= 0.28:
        candidates[tuple(plain_top)] += 0.06
    else:
        candidates[tuple(repel_light)] += 0.03
    best_key = max(candidates.items(), key=lambda kv: kv[1])[0]
    win_numbers = list(best_key)

    addl_model = normalize_prob(pred["addl"][0])
    combined_addl = (
        weights["addl_model"] * addl_model
        + weights["addl_cluster"] * normalize_prob(addl_cluster_prior)
    )
    combined_addl = normalize_prob(np.clip(combined_addl, 1e-12, None))

    addl_number = None
    for idx in np.argsort(combined_addl)[::-1]:
        candidate = int(idx + 1)
        if candidate not in win_numbers:
            addl_number = candidate
            break
    if addl_number is None:
        addl_number = int(np.argmax(combined_addl) + 1)
    return win_numbers, addl_number


def summarize_hits(pred_win: List[int], pred_addl: int, actual_win: List[int], actual_addl: int) -> Tuple[int, int]:
    hit_count = len(set(pred_win).intersection(actual_win))
    addl_hit = 1 if pred_addl == actual_addl else 0
    return hit_count, addl_hit


def recency_weights_from_draw_date(
    draw_values: np.ndarray,
    date_values: np.ndarray,
    order_values: np.ndarray | None = None,
) -> np.ndarray:
    n = int(len(draw_values))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    draw_num = pd.to_numeric(pd.Series(draw_values), errors="coerce").astype(float).values
    date_ser = pd.to_datetime(pd.Series(date_values), errors="coerce")
    date_num = date_ser.astype("int64", copy=False).values.astype(np.float64)
    bad_dt = ~np.isfinite(date_num) | (date_num < 0)
    if np.any(bad_dt):
        fill = float(np.nanmedian(date_num[~bad_dt])) if np.any(~bad_dt) else 0.0
        date_num = date_num.copy()
        date_num[bad_dt] = fill

    if order_values is None:
        order_base = np.arange(n, dtype=np.float64)
    else:
        order_base = np.asarray(order_values, dtype=np.float64)
        if len(order_base) != n:
            order_base = np.arange(n, dtype=np.float64)

    def _to_unit(v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=np.float64)
        bad = ~np.isfinite(x)
        if np.any(bad):
            x = x.copy()
            med = float(np.nanmedian(x[~bad])) if np.any(~bad) else 0.0
            x[bad] = med
        lo = float(np.min(x))
        hi = float(np.max(x))
        if abs(hi - lo) < 1e-12:
            return np.zeros_like(x, dtype=np.float64)
        return (x - lo) / (hi - lo + 1e-12)

    draw_u = _to_unit(draw_num)
    date_u = _to_unit(date_num)
    order_u = _to_unit(order_base)
    recency = 0.35 * draw_u + 0.35 * date_u + 0.30 * order_u
    weights = 0.28 + 1.72 * np.power(np.clip(recency, 0.0, 1.0), 2.2)
    weights = weights / (float(np.mean(weights)) + 1e-12)
    return weights.astype(np.float32)


def recency_weights_for_target_rows(df: pd.DataFrame, target_row_ids: np.ndarray) -> np.ndarray:
    if len(target_row_ids) <= 0:
        return np.zeros(0, dtype=np.float32)
    row_ids = np.asarray(target_row_ids, dtype=np.int32)
    draw_vals = df.loc[row_ids, "Draw"].values
    date_vals = df.loc[row_ids, "Date"].values
    order_vals = np.arange(len(row_ids), dtype=np.float64)
    return recency_weights_from_draw_date(draw_vals, date_vals, order_vals)


def sharpen_recency_weights(weights: np.ndarray, power: float = 1.35, floor: float = 0.32) -> np.ndarray:
    """
    Raise recency contrast for training sample weighting while keeping stability.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) == 0:
        return w.astype(np.float32)
    w = np.clip(w, 1e-8, None)
    w = np.power(w, max(1.0, float(power)))
    w = w / (float(np.mean(w)) + 1e-12)
    w = np.clip(w, float(floor), 4.5)
    return w.astype(np.float32)


def evaluate_hit_score(
    model: keras.Model,
    scaler_x: StandardScaler,
    X_seq: np.ndarray,
    eval_idx: np.ndarray,
    target_rows: np.ndarray,
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    weights: Dict[str, float],
    sample_weights: np.ndarray | None = None,
    cached_preds: Dict[str, np.ndarray] | None = None,
) -> Dict[str, float]:
    if len(eval_idx) == 0:
        return {
            "avg_win_hits": 0.0,
            "addl_acc": 0.0,
            "p_hit_ge2": 0.0,
            "p_hit_ge3": 0.0,
            "p_hit_ge4": 0.0,
            "hit_score": 0.0,
        }

    if cached_preds is None:
        n_features = X_seq.shape[-1]
        X_eval = scaler_x.transform(X_seq[eval_idx].reshape(-1, n_features)).reshape(
            len(eval_idx), X_seq.shape[1], n_features
        ).astype(np.float32)
        preds = predict_outputs_dict(model, X_eval)
    else:
        preds = cached_preds

    win_hits: List[int] = []
    addl_hits: List[int] = []
    for local_i, seq_i in enumerate(eval_idx):
        target_row = int(target_rows[seq_i])
        pred_pack = {k: v[local_i : local_i + 1] for k, v in preds.items()}

        pred_win, pred_addl = combine_prediction(
            pred=pred_pack,
            win_cluster_prior=win_cluster_prior_matrix[target_row],
            addl_cluster_prior=addl_cluster_prior_matrix[target_row],
            repel_prior=repel_prior_matrix[target_row],
            weights=weights,
        )
        actual_win = df.loc[target_row, WIN_COLS].astype(int).tolist()
        actual_addl = int(df.loc[target_row, "Addl No."])
        h, a = summarize_hits(pred_win, pred_addl, actual_win, actual_addl)
        win_hits.append(h)
        addl_hits.append(a)

    win_hits_arr = np.asarray(win_hits, dtype=np.float32)
    addl_arr = np.asarray(addl_hits, dtype=np.float32)
    return hit_metrics_from_arrays(win_hits_arr, addl_arr, sample_weights=sample_weights)


def hit_metrics_from_arrays(
    win_hits: np.ndarray,
    addl_hits: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> Dict[str, float]:
    if len(win_hits) == 0:
        return {
            "avg_win_hits": 0.0,
            "addl_acc": 0.0,
            "p_hit_ge2": 0.0,
            "p_hit_ge3": 0.0,
            "p_hit_ge4": 0.0,
            "hit_score": 0.0,
        }
    win_hits_arr = np.asarray(win_hits, dtype=np.float32)
    addl_arr = np.asarray(addl_hits, dtype=np.float32)
    if sample_weights is None:
        w = np.ones(len(win_hits_arr), dtype=np.float64)
    else:
        w = np.asarray(sample_weights, dtype=np.float64)
        if len(w) != len(win_hits_arr):
            w = np.ones(len(win_hits_arr), dtype=np.float64)
    w = np.clip(w, 1e-8, None)
    w = w / (float(np.mean(w)) + 1e-12)
    wsum = float(np.sum(w)) + 1e-12

    avg_win_hits = float(np.dot(w, win_hits_arr) / wsum)
    addl_acc = float(np.dot(w, addl_arr) / wsum)
    p_hit_ge2 = float(np.dot(w, (win_hits_arr >= 2).astype(np.float32)) / wsum)
    p_hit_ge3 = float(np.dot(w, (win_hits_arr >= 3).astype(np.float32)) / wsum)
    p_hit_ge4 = float(np.dot(w, (win_hits_arr >= 4).astype(np.float32)) / wsum)
    hit_score = 0.75 * avg_win_hits + 12.0 * p_hit_ge2 + 28.0 * p_hit_ge3 + 70.0 * p_hit_ge4 + 2.2 * addl_acc
    return {
        "avg_win_hits": avg_win_hits,
        "addl_acc": addl_acc,
        "p_hit_ge2": p_hit_ge2,
        "p_hit_ge3": p_hit_ge3,
        "p_hit_ge4": p_hit_ge4,
        "hit_score": hit_score,
    }


def reward_guided_refinement(
    model: keras.Model,
    scaler_x: StandardScaler,
    aux_scaler: StandardScaler,
    X_seq: np.ndarray,
    y_raw: Dict[str, np.ndarray],
    target_rows: np.ndarray,
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    blend_weights: Dict[str, float],
    reward_window: int,
    min_samples: int,
    epochs: int,
    batch_size: int,
) -> Dict[str, float]:
    summary: Dict[str, float] = {
        "enabled": 0.0,
        "reverted": 0.0,
        "samples": 0.0,
        "epochs": 0.0,
        "mean_reward": 0.0,
        "max_reward": 0.0,
        "pre_hit_score": 0.0,
        "post_hit_score": 0.0,
        "delta_hit_score": 0.0,
        "pre_p_hit_ge3": 0.0,
        "post_p_hit_ge3": 0.0,
        "pre_p_hit_ge4": 0.0,
        "post_p_hit_ge4": 0.0,
        "pre_avg_hits": 0.0,
        "post_avg_hits": 0.0,
        "loss_start": 0.0,
        "loss_end": 0.0,
    }
    if epochs <= 0 or len(X_seq) <= 0:
        return summary

    reward_idx = recent_seq_indices(
        len(X_seq),
        max(reward_window, min_samples),
        min_train=max(96, X_seq.shape[1] * 3),
    )
    if len(reward_idx) < min_samples:
        start = max(0, len(X_seq) - max(min_samples, reward_window))
        reward_idx = np.arange(start, len(X_seq), dtype=np.int32)
    if len(reward_idx) < max(12, min_samples // 2):
        return summary

    n_features = X_seq.shape[-1]
    X_reward = scaler_x.transform(X_seq[reward_idx].reshape(-1, n_features)).reshape(
        len(reward_idx), X_seq.shape[1], n_features
    ).astype(np.float32)

    y_reward: Dict[str, np.ndarray] = {}
    for key, arr in y_raw.items():
        if key == "aux":
            y_reward[key] = aux_scaler.transform(arr[reward_idx]).astype(np.float32)
        elif key == "win_set":
            y_reward[key] = arr[reward_idx].astype(np.float32)
        else:
            y_reward[key] = arr[reward_idx].astype(np.int32)

    preds = predict_outputs_dict(model, X_reward)
    win_hits: List[int] = []
    addl_hits: List[int] = []
    conf_scores: List[float] = []
    for local_i, seq_i in enumerate(reward_idx):
        target_row = int(target_rows[seq_i])
        pred_pack = {k: v[local_i : local_i + 1] for k, v in preds.items()}
        pred_win, pred_addl = combine_prediction(
            pred=pred_pack,
            win_cluster_prior=win_cluster_prior_matrix[target_row],
            addl_cluster_prior=addl_cluster_prior_matrix[target_row],
            repel_prior=repel_prior_matrix[target_row],
            weights=blend_weights,
        )
        actual_win = df.loc[target_row, WIN_COLS].astype(int).tolist()
        actual_addl = int(df.loc[target_row, "Addl No."])
        h, a = summarize_hits(pred_win, pred_addl, actual_win, actual_addl)
        win_hits.append(h)
        addl_hits.append(a)

        win_set_vec = pred_pack["win_set"][0].astype(np.float64)
        top6 = np.sort(win_set_vec)[-6:]
        conf_scores.append(float(np.mean(top6)))

    win_hits_arr = np.asarray(win_hits, dtype=np.float32)
    addl_arr = np.asarray(addl_hits, dtype=np.float32)
    conf_arr = np.asarray(conf_scores, dtype=np.float32)
    pre_metrics = hit_metrics_from_arrays(win_hits_arr, addl_arr)

    # Reward gives higher emphasis to multi-hit samples (>=3 and >=4) and addl hits.
    rewards = (
        0.55
        + 0.18 * win_hits_arr
        + 0.85 * (win_hits_arr >= 3).astype(np.float32)
        + 1.55 * (win_hits_arr >= 4).astype(np.float32)
        + 0.40 * addl_arr
        + 0.28 * np.clip(conf_arr, 0.0, 1.0)
    )
    rewards = np.clip(rewards, 0.20, 6.0)
    rewards = rewards / (float(np.mean(rewards)) + 1e-8)
    rewards = np.clip(rewards, 0.20, 3.80).astype(np.float32)

    sample_weight = {
        "win_1": rewards,
        "win_2": rewards,
        "win_3": rewards,
        "win_4": rewards,
        "win_5": rewards,
        "win_6": rewards,
        "win_set": rewards,
        "addl": rewards,
        "aux": rewards,
    }
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, min_lr=1e-5, verbose=0),
    ]
    pre_weights = model.get_weights()
    history = model.fit(
        X_reward,
        y_reward,
        sample_weight=sample_weight,
        batch_size=max(16, min(batch_size, len(X_reward))),
        epochs=int(max(1, epochs)),
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )

    post_metrics = evaluate_hit_score(
        model=model,
        scaler_x=scaler_x,
        X_seq=X_seq,
        eval_idx=reward_idx,
        target_rows=target_rows,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        weights=blend_weights,
    )
    losses = history.history.get("loss", [])

    pre_tail_obj = blend_objective(pre_metrics, focused=True)
    post_tail_obj = blend_objective(post_metrics, focused=True)
    reverted = 0.0
    # Keep reward update only when it improves tail-hit objective.
    if post_tail_obj + 1e-10 < pre_tail_obj:
        model.set_weights(pre_weights)
        post_metrics = dict(pre_metrics)
        reverted = 1.0

    summary.update(
        {
            "enabled": 1.0,
            "reverted": float(reverted),
            "samples": float(len(reward_idx)),
            "epochs": float(len(losses)),
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "pre_hit_score": float(pre_metrics["hit_score"]),
            "post_hit_score": float(post_metrics["hit_score"]),
            "delta_hit_score": float(post_metrics["hit_score"] - pre_metrics["hit_score"]),
            "pre_p_hit_ge3": float(pre_metrics["p_hit_ge3"]),
            "post_p_hit_ge3": float(post_metrics["p_hit_ge3"]),
            "pre_p_hit_ge4": float(pre_metrics["p_hit_ge4"]),
            "post_p_hit_ge4": float(post_metrics["p_hit_ge4"]),
            "pre_avg_hits": float(pre_metrics["avg_win_hits"]),
            "post_avg_hits": float(post_metrics["avg_win_hits"]),
            "loss_start": float(losses[0]) if losses else 0.0,
            "loss_end": float(losses[-1]) if losses else 0.0,
        }
    )
    return summary


def generate_trial_configs(num_trials: int) -> List[ModelConfig]:
    # Anchor from best observed run: log_01_s42_A (avg_win_hits=1.7000).
    anchor = ModelConfig(seq_len=24, lstm_units=128, gru_units=64, dense_units=160, dropout=0.2, lr=3e-4)
    seq_lens = [18, 24, 30, 36]
    lstm_units = [96, 128]
    gru_units = [64, 96]
    dense_units = [128, 160, 192]
    dropouts = [0.2, 0.28, 0.35]
    lrs = [3e-4, 2e-4, 1.5e-4]

    all_configs = [
        ModelConfig(
            seq_len=s,
            lstm_units=lstm,
            gru_units=gru,
            dense_units=dense,
            dropout=drop,
            lr=lr,
        )
        for s, lstm, gru, dense, drop, lr in itertools.product(seq_lens, lstm_units, gru_units, dense_units, dropouts, lrs)
    ]
    random.shuffle(all_configs)
    target_n = max(1, int(num_trials))
    out: List[ModelConfig] = [anchor]
    for cfg in all_configs:
        if len(out) >= target_n:
            break
        if (
            cfg.seq_len == anchor.seq_len
            and cfg.lstm_units == anchor.lstm_units
            and cfg.gru_units == anchor.gru_units
            and cfg.dense_units == anchor.dense_units
            and abs(cfg.dropout - anchor.dropout) < 1e-12
            and abs(cfg.lr - anchor.lr) < 1e-12
        ):
            continue
        out.append(cfg)
    return out


def generate_blend_candidates(base: Dict[str, float]) -> List[Dict[str, float]]:
    cands: List[Dict[str, float]] = []
    win_total = float(sum(base[k] for k in WIN_BLEND_KEYS))
    addl_total = float(sum(base[k] for k in ADDL_BLEND_KEYS))
    cands.append(normalize_blend_groups(dict(base), win_total=win_total, addl_total=addl_total))

    variants = [
        # model-heavy
        {"model_soft": 0.74, "model_set": 0.86, "cluster": 0.20, "repel": 0.04, "addl_model": 0.92, "addl_cluster": 0.20},
        # cluster-heavy
        {"model_soft": 0.44, "model_set": 0.52, "cluster": 0.84, "repel": 0.08, "addl_model": 0.56, "addl_cluster": 0.56},
        # balanced
        {"model_soft": 0.56, "model_set": 0.62, "cluster": 0.62, "repel": 0.07, "addl_model": 0.74, "addl_cluster": 0.38},
        # low repel
        {"model_soft": 0.50, "model_set": 0.56, "cluster": 0.74, "repel": 0.02, "addl_model": 0.68, "addl_cluster": 0.44},
        # high model_set
        {"model_soft": 0.34, "model_set": 1.14, "cluster": 0.32, "repel": 0.06, "addl_model": 0.82, "addl_cluster": 0.30},
        # high cluster + repel guard
        {"model_soft": 0.38, "model_set": 0.50, "cluster": 0.92, "repel": 0.12, "addl_model": 0.60, "addl_cluster": 0.52},
    ]
    for v in variants:
        cands.append(normalize_blend_groups(dict(v), win_total=win_total, addl_total=addl_total))
    return cands


def generate_backtest_blend_candidates(base: Dict[str, float], num_random: int = 220) -> List[Dict[str, float]]:
    """
    Generate a larger local search space around current blend weights.
    Used to maximize recent-window hit behavior without retraining models.
    """
    cands = generate_blend_candidates(base)
    rng = np.random.default_rng(ACTIVE_SEED + 317)
    win_keys = list(WIN_BLEND_KEYS)
    addl_keys = list(ADDL_BLEND_KEYS)
    win_total = float(sum(base[k] for k in win_keys))
    addl_total = float(sum(base[k] for k in addl_keys))
    base_norm = normalize_blend_groups(dict(base), win_total=win_total, addl_total=addl_total)
    base_win_prob = normalize_prob(np.asarray([base_norm[k] for k in win_keys], dtype=np.float64))
    base_addl_prob = normalize_prob(np.asarray([base_norm[k] for k in addl_keys], dtype=np.float64))

    for _ in range(max(1, num_random)):
        cand: Dict[str, float] = {}
        if rng.random() < 0.56:
            win_jitter = np.exp(rng.normal(0.0, 0.45, size=len(win_keys)))
            addl_jitter = np.exp(rng.normal(0.0, 0.42, size=len(addl_keys)))
            win_prob = normalize_prob(base_win_prob * win_jitter)
            addl_prob = normalize_prob(base_addl_prob * addl_jitter)
        else:
            win_alpha = np.clip(base_win_prob * 9.0, 0.07, None)
            addl_alpha = np.clip(base_addl_prob * 7.0, 0.10, None)
            if rng.random() < 0.45:
                win_alpha *= 0.52
            if rng.random() < 0.35:
                addl_alpha *= 0.60
            win_prob = rng.dirichlet(win_alpha)
            addl_prob = rng.dirichlet(addl_alpha)

        for i, k in enumerate(win_keys):
            cand[k] = float(win_prob[i] * win_total)
        for i, k in enumerate(addl_keys):
            cand[k] = float(addl_prob[i] * addl_total)

        repel_noise = rng.normal(0.0, 0.032)
        cand["repel"] = float(np.clip(base_norm["repel"] + repel_noise, 0.0, 0.24))
        if rng.random() < 0.18:
            cand["repel"] = float(np.clip(cand["repel"] * 0.35, 0.0, 0.24))
        cands.append(normalize_blend_groups(cand, win_total=win_total, addl_total=addl_total))

    # Add explicit concentrated candidates to probe high-hit tails.
    dominant_win = [k for k in win_keys[:]]
    dominant_win.sort(key=lambda k: base_norm[k], reverse=True)
    support_key = dominant_win[1] if len(dominant_win) > 1 else dominant_win[0]
    for top_key in dominant_win[:3]:
        cand = {k: 1e-4 for k in win_keys + addl_keys}
        cand[top_key] = 0.78 * win_total
        cand[support_key] += 0.22 * win_total
        for k in addl_keys:
            cand[k] = float(base_norm[k])
        cand["repel"] = float(np.clip(base_norm["repel"] * 0.45, 0.0, 0.24))
        cands.append(normalize_blend_groups(cand, win_total=win_total, addl_total=addl_total))
    return cands


def blend_objective(metrics: Dict[str, float], focused: bool = False) -> float:
    p2 = float(metrics.get("p_hit_ge2", 0.0))
    p3 = float(metrics.get("p_hit_ge3", 0.0))
    p4 = float(metrics.get("p_hit_ge4", 0.0))
    avg = float(metrics.get("avg_win_hits", 0.0))
    addl = float(metrics.get("addl_acc", 0.0))
    if focused:
        tail_cross = p4 * p3
        tail_quad = (p4 ** 2) + 0.55 * (p3 ** 2)
        gate_bonus = 30.0 * float(p4 > 0.0) + 12.0 * float(p3 >= 0.20)
        return (
            620.0 * p4
            + 245.0 * p3
            + 126.0 * p2
            + 16.5 * avg
            + 2.8 * addl
            + 210.0 * tail_cross
            + 140.0 * tail_quad
            + gate_bonus
        )
    return 300.0 * p4 + 125.0 * p3 + 70.0 * p2 + 8.0 * avg + 2.2 * addl


def normalize_blend_groups(
    weights: Dict[str, float],
    win_total: float,
    addl_total: float,
    min_value: float = 1e-4,
) -> Dict[str, float]:
    out = dict(weights)
    win_vals = np.asarray([max(min_value, float(out[k])) for k in WIN_BLEND_KEYS], dtype=np.float64)
    addl_vals = np.asarray([max(min_value, float(out[k])) for k in ADDL_BLEND_KEYS], dtype=np.float64)
    win_vals *= float(max(min_value, win_total)) / float(np.sum(win_vals) + 1e-12)
    addl_vals *= float(max(min_value, addl_total)) / float(np.sum(addl_vals) + 1e-12)
    for i, k in enumerate(WIN_BLEND_KEYS):
        out[k] = float(win_vals[i])
    for i, k in enumerate(ADDL_BLEND_KEYS):
        out[k] = float(addl_vals[i])
    out["repel"] = float(np.clip(float(out["repel"]), 0.0, 0.24))
    return out


def coordinate_ascent_blend_search(
    start_weights: Dict[str, float],
    eval_fn: Callable[[Dict[str, float]], Dict[str, float]],
    focused: bool,
    max_iters: int = 7,
    init_step: float = 0.34,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    best = dict(start_weights)
    win_total = float(sum(best[k] for k in WIN_BLEND_KEYS))
    addl_total = float(sum(best[k] for k in ADDL_BLEND_KEYS))
    best = normalize_blend_groups(best, win_total=win_total, addl_total=addl_total)
    best_metrics = eval_fn(best)
    best_score = float(blend_objective(best_metrics, focused=focused))

    key_order = WIN_BLEND_KEYS + ADDL_BLEND_KEYS + ["repel"]
    step = float(np.clip(init_step, 0.08, 0.65))
    min_step = 0.02
    score_eps = 1e-9
    eval_cache: Dict[Tuple[float, ...], Tuple[Dict[str, float], float]] = {}

    def candidate_key(w: Dict[str, float]) -> Tuple[float, ...]:
        keys = WIN_BLEND_KEYS + ADDL_BLEND_KEYS + ["repel"]
        return tuple(round(float(w[k]), 8) for k in keys)

    for _ in range(max(1, max_iters)):
        improved_round = False
        for k in key_order:
            local_best = best
            local_metrics = best_metrics
            local_score = best_score
            for direction in (-1.0, 1.0):
                trial = dict(best)
                if k == "repel":
                    trial[k] = float(np.clip(best[k] + direction * step * 0.08, 0.0, 0.24))
                else:
                    base_v = max(1e-4, float(best[k]))
                    trial[k] = float(max(1e-4, base_v * (1.0 + direction * step)))
                    if direction > 0.0 and base_v < 0.002:
                        floor_push = 0.003 if k in WIN_BLEND_KEYS else 0.006
                        trial[k] = max(trial[k], floor_push)
                trial = normalize_blend_groups(trial, win_total=win_total, addl_total=addl_total)
                ck = candidate_key(trial)
                if ck in eval_cache:
                    trial_metrics, trial_score = eval_cache[ck]
                else:
                    trial_metrics = eval_fn(trial)
                    trial_score = float(blend_objective(trial_metrics, focused=focused))
                    eval_cache[ck] = (trial_metrics, trial_score)

                if trial_score > local_score + score_eps:
                    local_best = trial
                    local_metrics = trial_metrics
                    local_score = trial_score

            if local_score > best_score + score_eps:
                best = local_best
                best_metrics = local_metrics
                best_score = local_score
                improved_round = True

        if not improved_round:
            step *= 0.62
        else:
            step *= 0.92
        if step < min_step:
            break

    return best, best_metrics, best_score


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-x))


def _inv_sigmoid(y: float) -> float:
    y = float(np.clip(y, 1e-6, 1.0 - 1e-6))
    return float(math.log(y / (1.0 - y)))


def _softmax_np(v: np.ndarray) -> np.ndarray:
    z = v.astype(np.float64) - float(np.max(v))
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 0:
        return np.ones_like(z, dtype=np.float64) / float(len(z))
    return e / s


def encode_blend_weights(weights: Dict[str, float]) -> np.ndarray:
    win_total = float(sum(weights[k] for k in WIN_BLEND_KEYS))
    addl_total = float(sum(weights[k] for k in ADDL_BLEND_KEYS))
    win_vec = np.asarray([max(1e-6, float(weights[k])) for k in WIN_BLEND_KEYS], dtype=np.float64)
    addl_vec = np.asarray([max(1e-6, float(weights[k])) for k in ADDL_BLEND_KEYS], dtype=np.float64)
    win_prob = win_vec / max(1e-9, win_total)
    addl_prob = addl_vec / max(1e-9, addl_total)
    win_logits = np.log(np.clip(win_prob, 1e-9, 1.0))
    addl_logits = np.log(np.clip(addl_prob, 1e-9, 1.0))
    repel_logit = _inv_sigmoid(float(weights["repel"]) / 0.24 if weights["repel"] > 0 else 1e-6)
    return np.concatenate([win_logits, addl_logits, np.asarray([repel_logit], dtype=np.float64)]).astype(np.float64)


def decode_blend_weights(
    vec: np.ndarray,
    win_total: float,
    addl_total: float,
) -> Dict[str, float]:
    v = np.asarray(vec, dtype=np.float64)
    win_logits = v[: len(WIN_BLEND_KEYS)]
    addl_logits = v[len(WIN_BLEND_KEYS) : len(WIN_BLEND_KEYS) + len(ADDL_BLEND_KEYS)]
    repel_logit = float(v[-1])

    win_prob = _softmax_np(win_logits)
    addl_prob = _softmax_np(addl_logits)
    out: Dict[str, float] = {}
    for i, k in enumerate(WIN_BLEND_KEYS):
        out[k] = float(win_prob[i] * max(1e-6, win_total))
    for i, k in enumerate(ADDL_BLEND_KEYS):
        out[k] = float(addl_prob[i] * max(1e-6, addl_total))
    out["repel"] = float(np.clip(_sigmoid(repel_logit) * 0.24, 0.0, 0.24))
    return normalize_blend_groups(out, win_total=win_total, addl_total=addl_total)


def nelder_mead_blend_search(
    start_weights: Dict[str, float],
    eval_fn: Callable[[Dict[str, float]], Dict[str, float]],
    focused: bool,
    max_iters: int = 56,
    step: float = 0.18,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    win_total = float(sum(start_weights[k] for k in WIN_BLEND_KEYS))
    addl_total = float(sum(start_weights[k] for k in ADDL_BLEND_KEYS))
    x0 = encode_blend_weights(start_weights)
    dim = len(x0)
    init_step = float(np.clip(step, 0.04, 0.65))

    simplex: List[np.ndarray] = [x0.copy()]
    for i in range(dim):
        xi = x0.copy()
        xi[i] += init_step
        simplex.append(xi)

    score_cache: Dict[Tuple[float, ...], Tuple[Dict[str, float], float]] = {}

    def eval_vec(x: np.ndarray) -> Tuple[Dict[str, float], float]:
        key = tuple(np.round(x.astype(np.float64), 8).tolist())
        if key in score_cache:
            return score_cache[key]
        w = decode_blend_weights(x, win_total=win_total, addl_total=addl_total)
        m = eval_fn(w)
        s = float(blend_objective(m, focused=focused))
        score_cache[key] = (m, s)
        return m, s

    for _ in range(max(1, int(max_iters))):
        evals = []
        for x in simplex:
            m, s = eval_vec(x)
            evals.append((x, m, s))
        evals.sort(key=lambda t: t[2], reverse=True)
        simplex = [t[0] for t in evals]
        best_x, best_m, best_s = evals[0]
        worst_x, _, worst_s = evals[-1]
        second_worst_s = evals[-2][2]

        centroid = np.mean(np.stack(simplex[:-1], axis=0), axis=0)
        xr = centroid + alpha * (centroid - worst_x)
        mr, sr = eval_vec(xr)

        if sr > best_s:
            xe = centroid + gamma * (xr - centroid)
            me, se = eval_vec(xe)
            simplex[-1] = xe if se > sr else xr
        elif sr > second_worst_s:
            simplex[-1] = xr
        else:
            if sr > worst_s:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid - rho * (xr - centroid)
            mc, sc = eval_vec(xc)
            if sc > worst_s:
                simplex[-1] = xc
            else:
                base = simplex[0]
                new_simplex = [base]
                for i in range(1, len(simplex)):
                    xs = base + sigma * (simplex[i] - base)
                    new_simplex.append(xs)
                simplex = new_simplex

        # Stop when simplex objective spread is tiny.
        evals_now = [eval_vec(x)[1] for x in simplex]
        if (max(evals_now) - min(evals_now)) < 1e-5:
            break

    final_evals = []
    for x in simplex:
        m, s = eval_vec(x)
        final_evals.append((x, m, s))
    final_evals.sort(key=lambda t: t[2], reverse=True)
    x_best, m_best, s_best = final_evals[0]
    w_best = decode_blend_weights(x_best, win_total=win_total, addl_total=addl_total)
    return w_best, m_best, float(s_best)


def tail_focus_blend_refine(
    start_weights: Dict[str, float],
    eval_fn: Callable[[Dict[str, float]], Dict[str, float]],
    max_iters: int = 4,
    candidates_per_iter: int = 160,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    rng = np.random.default_rng(ACTIVE_SEED + 887)
    win_total = float(sum(start_weights[k] for k in WIN_BLEND_KEYS))
    addl_total = float(sum(start_weights[k] for k in ADDL_BLEND_KEYS))

    best = normalize_blend_groups(dict(start_weights), win_total=win_total, addl_total=addl_total)
    best_metrics = eval_fn(best)
    best_obj = float(blend_objective(best_metrics, focused=True))
    eval_cache: Dict[Tuple[float, ...], Tuple[Dict[str, float], float]] = {}

    def _key(w: Dict[str, float]) -> Tuple[float, ...]:
        keys = WIN_BLEND_KEYS + ADDL_BLEND_KEYS + ["repel"]
        return tuple(round(float(w[k]), 8) for k in keys)

    def _is_better(lhs: Dict[str, float], rhs: Dict[str, float]) -> bool:
        for k in ("p_hit_ge4", "p_hit_ge3", "avg_win_hits", "p_hit_ge2", "addl_acc", "hit_score"):
            dv = float(lhs.get(k, 0.0)) - float(rhs.get(k, 0.0))
            if dv > 1e-10:
                return True
            if dv < -1e-10:
                return False
        return False

    def _eval(w: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        wk = _key(w)
        if wk in eval_cache:
            return eval_cache[wk]
        m = eval_fn(w)
        s = float(blend_objective(m, focused=True))
        eval_cache[wk] = (m, s)
        return m, s

    step = 0.36
    for _ in range(max(1, int(max_iters))):
        local_best = dict(best)
        local_metrics = dict(best_metrics)
        local_obj = float(best_obj)

        for _ in range(max(8, int(candidates_per_iter))):
            cand = dict(local_best)
            win_noise = np.exp(rng.normal(0.0, step, size=len(WIN_BLEND_KEYS)))
            addl_noise = np.exp(rng.normal(0.0, step * 0.82, size=len(ADDL_BLEND_KEYS)))
            for i, k in enumerate(WIN_BLEND_KEYS):
                cand[k] = max(1e-4, float(cand[k] * win_noise[i]))
            for i, k in enumerate(ADDL_BLEND_KEYS):
                cand[k] = max(1e-4, float(cand[k] * addl_noise[i]))
            cand["repel"] = float(np.clip(cand["repel"] + rng.normal(0.0, 0.024 + 0.04 * step), 0.0, 0.24))
            cand = normalize_blend_groups(cand, win_total=win_total, addl_total=addl_total)

            m, s = _eval(cand)
            if _is_better(m, local_metrics) or (not _is_better(local_metrics, m) and s > local_obj + 1e-10):
                local_best = cand
                local_metrics = m
                local_obj = s

        if _is_better(local_metrics, best_metrics) or (
            not _is_better(best_metrics, local_metrics) and local_obj > best_obj + 1e-10
        ):
            best = local_best
            best_metrics = local_metrics
            best_obj = local_obj
        step = max(0.06, step * 0.62)

    return best, best_metrics, best_obj


def squash_component_scores(
    scores: np.ndarray,
    cutoff: float = 0.36,
    power: float = 4.0,
    floor: float = 1e-4,
) -> np.ndarray:
    s = scores.astype(np.float64)
    if len(s) == 0:
        return s
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    if abs(s_max - s_min) < 1e-10:
        return np.ones_like(s, dtype=np.float64) / float(len(s))
    q = (s - s_min) / (s_max - s_min + 1e-12)
    z = np.full_like(q, fill_value=floor, dtype=np.float64)
    mask = q >= cutoff
    if np.any(mask):
        z[mask] = np.power(np.clip((q[mask] - cutoff) / max(1e-6, (1.0 - cutoff)), 0.0, 1.0), power)
    z = np.clip(z, floor, None)
    z_sum = float(z.sum())
    if z_sum <= 0:
        z = np.ones_like(z, dtype=np.float64)
        z_sum = float(z.sum())
    return z / z_sum


def run_tuning(
    trial_configs: List[ModelConfig],
    seq_cache: Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]],
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    blend_weights: Dict[str, float],
    tune_folds: int,
    tune_epochs: int,
    focus_last_n: int,
    batch_size: int,
    hit_score_focused: bool = False,
    steps_per_execution: int = 0,
    cache_dataset: bool = False,
    train_recency_weighted: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for trial_idx, cfg in enumerate(trial_configs, start=1):
        X_seq, y_raw, target_rows = seq_cache[cfg.seq_len]
        fold_losses: List[float] = []
        fold_hit_scores: List[float] = []
        fold_avg_hits: List[float] = []
        fold_p_ge2: List[float] = []
        fold_p_ge3: List[float] = []
        fold_p_ge4: List[float] = []
        fold_addl_acc: List[float] = []
        min_train = max(120, cfg.seq_len * 4)

        if focus_last_n > 0:
            # Recent-focus objective: tune on fold blocks carved only from the last N draws.
            recent_idx = recent_seq_indices(len(X_seq), focus_last_n, min_train=min_train)
            if len(recent_idx) > 0:
                block_size = max(2, int(math.ceil(len(recent_idx) / max(1, tune_folds))))
            else:
                block_size = 0
            used_blocks = 0
            for block_start in range(0, len(recent_idx), max(1, block_size)):
                if used_blocks >= max(1, tune_folds):
                    break
                eval_block = recent_idx[block_start : block_start + block_size]
                if len(eval_block) == 0:
                    continue
                seq_anchor = int(eval_block[0])
                train_idx, val_idx = one_step_train_val_indices(seq_anchor, min_train_core=min_train)
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue
                if bool(train_recency_weighted):
                    train_sw = sharpen_recency_weights(
                        recency_weights_for_target_rows(df, target_rows[np.asarray(train_idx, dtype=np.int32)]),
                        power=1.35,
                        floor=0.32,
                    )
                    val_sw = recency_weights_for_target_rows(df, target_rows[np.asarray(val_idx, dtype=np.int32)])
                else:
                    train_sw = None
                    val_sw = None
                model, history, _, scaler_x = run_single_fit(
                    config=cfg,
                    X_seq=X_seq,
                    y_raw=y_raw,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    epochs=tune_epochs,
                    batch_size=batch_size,
                    steps_per_execution=steps_per_execution,
                    cache_dataset=cache_dataset,
                    train_sample_weights=train_sw,
                    val_sample_weights=val_sw,
                    verbose=0,
                )
                best_fold_val = float(np.min(history.history.get("val_loss", [math.inf])))
                fold_losses.append(best_fold_val)

                eval_weights = recency_weights_for_target_rows(df, target_rows[np.asarray(eval_block, dtype=np.int32)])
                hit_metrics = evaluate_hit_score(
                    model=model,
                    scaler_x=scaler_x,
                    X_seq=X_seq,
                    eval_idx=np.asarray(eval_block, dtype=np.int32),
                    target_rows=target_rows,
                    df=df,
                    win_cluster_prior_matrix=win_cluster_prior_matrix,
                    addl_cluster_prior_matrix=addl_cluster_prior_matrix,
                    repel_prior_matrix=repel_prior_matrix,
                    weights=blend_weights,
                    sample_weights=eval_weights,
                )
                fold_hit_scores.append(hit_metrics["hit_score"])
                fold_avg_hits.append(hit_metrics["avg_win_hits"])
                fold_p_ge2.append(hit_metrics["p_hit_ge2"])
                fold_p_ge3.append(hit_metrics["p_hit_ge3"])
                fold_p_ge4.append(hit_metrics["p_hit_ge4"])
                fold_addl_acc.append(hit_metrics["addl_acc"])
                used_blocks += 1
        else:
            splits = make_walk_forward_splits(len(X_seq), tune_folds)
            for train_idx, val_idx in splits:
                model, history, _, scaler_x = run_single_fit(
                    config=cfg,
                    X_seq=X_seq,
                    y_raw=y_raw,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    epochs=tune_epochs,
                    batch_size=batch_size,
                    steps_per_execution=steps_per_execution,
                    cache_dataset=cache_dataset,
                    verbose=0,
                )
                best_fold_val = float(np.min(history.history.get("val_loss", [math.inf])))
                fold_losses.append(best_fold_val)

                hit_metrics = evaluate_hit_score(
                    model=model,
                    scaler_x=scaler_x,
                    X_seq=X_seq,
                    eval_idx=val_idx,
                    target_rows=target_rows,
                    df=df,
                    win_cluster_prior_matrix=win_cluster_prior_matrix,
                    addl_cluster_prior_matrix=addl_cluster_prior_matrix,
                    repel_prior_matrix=repel_prior_matrix,
                    weights=blend_weights,
                )
                fold_hit_scores.append(hit_metrics["hit_score"])
                fold_avg_hits.append(hit_metrics["avg_win_hits"])
                fold_p_ge2.append(hit_metrics["p_hit_ge2"])
                fold_p_ge3.append(hit_metrics["p_hit_ge3"])
                fold_p_ge4.append(hit_metrics["p_hit_ge4"])
                fold_addl_acc.append(hit_metrics["addl_acc"])

        mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        mean_hit_score = float(np.mean(fold_hit_scores)) if fold_hit_scores else -1.0
        mean_avg_hits = float(np.mean(fold_avg_hits)) if fold_avg_hits else 0.0
        mean_p_ge2 = float(np.mean(fold_p_ge2)) if fold_p_ge2 else 0.0
        mean_p_ge3 = float(np.mean(fold_p_ge3)) if fold_p_ge3 else 0.0
        mean_p_ge4 = float(np.mean(fold_p_ge4)) if fold_p_ge4 else 0.0
        mean_addl_acc = float(np.mean(fold_addl_acc)) if fold_addl_acc else 0.0
        tune_obj = blend_objective(
            {
                "p_hit_ge2": mean_p_ge2,
                "p_hit_ge3": mean_p_ge3,
                "p_hit_ge4": mean_p_ge4,
                "avg_win_hits": mean_avg_hits,
                "addl_acc": mean_addl_acc,
            },
            focused=bool(hit_score_focused),
        )
        rows.append(
            {
                "trial": trial_idx,
                "seq_len": cfg.seq_len,
                "lstm_units": cfg.lstm_units,
                "gru_units": cfg.gru_units,
                "dense_units": cfg.dense_units,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "mean_val_loss": mean_loss,
                "mean_val_hit_score": mean_hit_score,
                "mean_val_avg_hits": mean_avg_hits,
                "mean_val_p_hit_ge2": mean_p_ge2,
                "mean_val_p_hit_ge3": mean_p_ge3,
                "mean_val_p_hit_ge4": mean_p_ge4,
                "mean_val_addl_acc": mean_addl_acc,
                "mean_val_tune_obj": float(tune_obj),
                "folds_used": len(fold_losses),
            }
        )
        print(
            f"[TUNE] Trial {trial_idx}/{len(trial_configs)} "
            f"seq={cfg.seq_len}, lstm={cfg.lstm_units}, gru={cfg.gru_units}, "
            f"drop={cfg.dropout}, lr={cfg.lr} -> p>=2={mean_p_ge2:.4f}, p>=3={mean_p_ge3:.4f}, "
            f"p>=4={mean_p_ge4:.4f}, avg_hits={mean_avg_hits:.4f}, val_loss={mean_loss:.4f}, "
            f"recent_mode={'Y' if focus_last_n > 0 else 'N'} folds={len(fold_losses)}"
        )

    result_df = pd.DataFrame(rows).sort_values(
        ["mean_val_tune_obj", "mean_val_p_hit_ge4", "mean_val_p_hit_ge3", "mean_val_avg_hits", "mean_val_loss"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return result_df


def calibrate_blend_weights_by_component_performance(
    model: keras.Model,
    scaler_x: StandardScaler,
    X_seq: np.ndarray,
    eval_idx: np.ndarray,
    target_rows: np.ndarray,
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    base_weights: Dict[str, float],
    sample_weights: np.ndarray | None = None,
    cached_preds: Dict[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    win_keys = list(WIN_BLEND_KEYS)
    addl_keys = list(ADDL_BLEND_KEYS)
    rows: List[Dict[str, object]] = []

    # Evaluate each win component independently against P(hit>=4), P(hit>=3), avg hits.
    win_scores: List[float] = []
    for key in win_keys:
        w = {k: 0.0 for k in base_weights.keys()}
        w[key] = 1.0
        w["addl_model"] = 1.0
        metrics = evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=w,
            sample_weights=sample_weights,
            cached_preds=cached_preds,
        )
        score = (
            320.0 * metrics["p_hit_ge4"]
            + 140.0 * metrics["p_hit_ge3"]
            + 90.0 * metrics["p_hit_ge2"]
            + 9.0 * metrics["avg_win_hits"]
        )
        win_scores.append(float(score))
        rows.append(
            {
                "component": key,
                "group": "win",
                "score": float(score),
                "p_hit_ge4": float(metrics["p_hit_ge4"]),
                "p_hit_ge2": float(metrics["p_hit_ge2"]),
                "p_hit_ge3": float(metrics["p_hit_ge3"]),
                "avg_win_hits": float(metrics["avg_win_hits"]),
                "addl_acc": float(metrics["addl_acc"]),
            }
        )

    # Evaluate each addl component independently by addl_acc with weak win stabilizer.
    addl_scores: List[float] = []
    for key in addl_keys:
        w = {k: 0.0 for k in base_weights.keys()}
        w["model_set"] = 0.55
        w["model_soft"] = 0.30
        w["cluster"] = 0.15
        w[key] = 1.0
        metrics = evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=w,
            sample_weights=sample_weights,
            cached_preds=cached_preds,
        )
        score = 28.0 * metrics["addl_acc"] + 2.0 * metrics["avg_win_hits"] + 18.0 * metrics["p_hit_ge2"]
        addl_scores.append(float(score))
        rows.append(
            {
                "component": key,
                "group": "addl",
                "score": float(score),
                "p_hit_ge4": float(metrics["p_hit_ge4"]),
                "p_hit_ge2": float(metrics["p_hit_ge2"]),
                "p_hit_ge3": float(metrics["p_hit_ge3"]),
                "avg_win_hits": float(metrics["avg_win_hits"]),
                "addl_acc": float(metrics["addl_acc"]),
            }
        )

    win_norm = squash_component_scores(np.asarray(win_scores, dtype=np.float64), cutoff=0.34, power=4.2, floor=1e-4)
    addl_norm = squash_component_scores(np.asarray(addl_scores, dtype=np.float64), cutoff=0.38, power=4.4, floor=1e-4)

    calibrated = dict(base_weights)
    win_total = sum(base_weights[k] for k in win_keys)
    addl_total = sum(base_weights[k] for k in addl_keys)
    for i, k in enumerate(win_keys):
        calibrated[k] = float(win_norm[i] * win_total)
    for i, k in enumerate(addl_keys):
        calibrated[k] = float(addl_norm[i] * addl_total)

    # Tune repel separately by direct objective; poor repel gets pushed near zero.
    best_repel = base_weights["repel"]
    best_repel_score = -1e18
    for r in [0.0, 0.02, 0.04, 0.07, 0.10, 0.14]:
        w = dict(calibrated)
        w["repel"] = float(r)
        metrics = evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=w,
            sample_weights=sample_weights,
            cached_preds=cached_preds,
        )
        score = blend_objective(metrics, focused=False)
        if score > best_repel_score:
            best_repel_score = float(score)
            best_repel = float(r)
    calibrated["repel"] = best_repel

    comp_df = pd.DataFrame(rows).sort_values(["group", "score"], ascending=[True, False]).reset_index(drop=True)
    return calibrated, comp_df


def run_backtest(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    best_k: int,
    config: ModelConfig,
    X_seq: np.ndarray,
    y_raw: Dict[str, np.ndarray],
    target_rows: np.ndarray,
    win_cluster_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    blend_weights: Dict[str, float],
    backtest_folds: int,
    backtest_epochs: int,
    focus_last_n: int,
    batch_size: int,
    local_random_candidates: int,
    optimize_avg: bool,
    backtest_restarts: int = 1,
    restart_ensemble_topk: int = 1,
    steps_per_execution: int = 0,
    cache_dataset: bool = False,
    train_recency_weighted: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n_samples = len(X_seq)
    records: List[Dict[str, object]] = []
    if focus_last_n > 0:
        # Strict one-step walk-forward on most recent N draws only.
        test_seq = recent_seq_indices(n_samples, focus_last_n, min_train=max(120, config.seq_len * 4))
        payloads: List[Dict[str, object]] = []
        for pos, seq_i in enumerate(test_seq, start=1):
            train_idx, val_idx = one_step_train_val_indices(int(seq_i), min_train_core=max(120, config.seq_len * 4))
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            n_features = X_seq.shape[-1]
            n_restart = int(max(1, backtest_restarts))
            top_k = int(max(1, min(restart_ensemble_topk, n_restart)))
            restart_pool: List[Tuple[float, Dict[str, np.ndarray]]] = []
            if bool(train_recency_weighted):
                val_sw = recency_weights_for_target_rows(df, target_rows[np.asarray(val_idx, dtype=np.int32)])
            else:
                val_sw = None

            for restart_i in range(n_restart):
                restart_seed = int(ACTIVE_SEED + 97 * restart_i + 13 * pos)
                set_global_seed(restart_seed)
                if bool(train_recency_weighted):
                    train_sw = sharpen_recency_weights(
                        recency_weights_for_target_rows(df, target_rows[np.asarray(train_idx, dtype=np.int32)]),
                        power=1.35,
                        floor=0.32,
                    )
                else:
                    train_sw = None
                model, _, _, scaler_x = run_single_fit(
                    config=config,
                    X_seq=X_seq,
                    y_raw=y_raw,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    epochs=backtest_epochs,
                    batch_size=batch_size,
                    steps_per_execution=steps_per_execution,
                    cache_dataset=cache_dataset,
                    train_sample_weights=train_sw,
                    val_sample_weights=val_sw,
                    verbose=0,
                )
                X_test = scaler_x.transform(X_seq[[seq_i]].reshape(-1, n_features)).reshape(
                    1, X_seq.shape[1], n_features
                ).astype(np.float32)
                pred_one = predict_outputs_dict(model, X_test)
                val_metrics = evaluate_hit_score(
                    model=model,
                    scaler_x=scaler_x,
                    X_seq=X_seq,
                    eval_idx=np.asarray(val_idx, dtype=np.int32),
                    target_rows=target_rows,
                    df=df,
                    win_cluster_prior_matrix=win_cluster_prior_matrix,
                    addl_cluster_prior_matrix=addl_cluster_prior_matrix,
                    repel_prior_matrix=repel_prior_matrix,
                    weights=blend_weights,
                    sample_weights=val_sw,
                )
                restart_score = float(
                    3.2 * val_metrics["avg_win_hits"]
                    + 2.8 * val_metrics["p_hit_ge3"]
                    + 7.2 * val_metrics["p_hit_ge4"]
                    + 0.3 * val_metrics["p_hit_ge2"]
                )
                restart_pool.append((restart_score, pred_one))

            restart_pool.sort(key=lambda x: x[0], reverse=True)
            chosen = restart_pool[:top_k]
            pred_pack = {
                k: np.mean(np.stack([p[k] for _, p in chosen], axis=0), axis=0)
                for k in chosen[0][1].keys()
            }
            target_row = int(target_rows[seq_i])

            hist_labels = cluster_labels[:target_row]
            _, last_cluster, next_cluster = transition_from_labels(hist_labels, best_k) if len(hist_labels) > 1 else (
                np.ones((best_k, best_k)) / best_k,
                int(cluster_labels[max(0, target_row - 1)]),
                int(cluster_labels[max(0, target_row - 1)]),
            )
            actual_win = df.loc[target_row, WIN_COLS].astype(int).tolist()
            actual_addl = int(df.loc[target_row, "Addl No."])
            payloads.append(
                {
                    "fold": pos,
                    "target_row": target_row,
                    "draw": int(df.loc[target_row, "Draw"]) if not pd.isna(df.loc[target_row, "Draw"]) else "",
                    "date": df.loc[target_row, "Date"].strftime("%Y-%m-%d") if not pd.isna(df.loc[target_row, "Date"]) else "",
                    "cluster_last": last_cluster,
                    "cluster_pred_next": next_cluster,
                    "pred_pack": pred_pack,
                    "win_cluster_prior": win_cluster_prior_matrix[target_row],
                    "addl_cluster_prior": addl_cluster_prior_matrix[target_row],
                    "repel_prior": repel_prior_matrix[target_row],
                    "actual_win": actual_win,
                    "actual_addl": actual_addl,
                }
            )

        if payloads:
            best_weights = dict(blend_weights)
            best_score = -1e12
            payload_draw = np.asarray([item["draw"] for item in payloads], dtype=object)
            payload_date = np.asarray([item["date"] for item in payloads], dtype=object)
            payload_order = np.arange(len(payloads), dtype=np.float64)
            payload_recency_w = recency_weights_from_draw_date(payload_draw, payload_date, payload_order)
            local_random = int(max(64, local_random_candidates if local_random_candidates > 0 else (560 if focus_last_n > 0 else 260)))
            for cand in generate_backtest_blend_candidates(blend_weights, num_random=local_random):
                hits: List[int] = []
                addl_hits: List[int] = []
                for item in payloads:
                    pred_win, pred_addl = combine_prediction(
                        pred=item["pred_pack"],
                        win_cluster_prior=item["win_cluster_prior"],
                        addl_cluster_prior=item["addl_cluster_prior"],
                        repel_prior=item["repel_prior"],
                        weights=cand,
                    )
                    h, a = summarize_hits(pred_win, pred_addl, item["actual_win"], int(item["actual_addl"]))
                    hits.append(h)
                    addl_hits.append(a)

                hits_arr = np.asarray(hits, dtype=np.float32)
                addl_arr = np.asarray(addl_hits, dtype=np.float32)
                m = hit_metrics_from_arrays(hits_arr, addl_arr, sample_weights=payload_recency_w)
                if bool(optimize_avg):
                    score = (
                        210.0 * float(m["avg_win_hits"])
                        + 150.0 * float(m["p_hit_ge2"])
                        + 95.0 * float(m["p_hit_ge3"])
                        + 52.0 * float(m["p_hit_ge4"])
                        + 6.0 * float(m["addl_acc"])
                    )
                else:
                    score = blend_objective(
                        {
                            "p_hit_ge2": float(m["p_hit_ge2"]),
                            "p_hit_ge3": float(m["p_hit_ge3"]),
                            "p_hit_ge4": float(m["p_hit_ge4"]),
                            "avg_win_hits": float(m["avg_win_hits"]),
                            "addl_acc": float(m["addl_acc"]),
                        },
                        focused=True,
                    )
                if score > best_score:
                    best_score = score
                    best_weights = dict(cand)

            for item in payloads:
                pred_win, pred_addl = combine_prediction(
                    pred=item["pred_pack"],
                    win_cluster_prior=item["win_cluster_prior"],
                    addl_cluster_prior=item["addl_cluster_prior"],
                    repel_prior=item["repel_prior"],
                    weights=best_weights,
                )
                win_hits, addl_hit = summarize_hits(pred_win, pred_addl, item["actual_win"], int(item["actual_addl"]))
                records.append(
                    {
                        "fold": item["fold"],
                        "target_row": item["target_row"],
                        "draw": item["draw"],
                        "date": item["date"],
                        "pred_win": " ".join(str(x) for x in pred_win),
                        "actual_win": " ".join(str(x) for x in sorted(item["actual_win"])),
                        "pred_addl": pred_addl,
                        "actual_addl": int(item["actual_addl"]),
                        "win_hits": win_hits,
                        "addl_hit": addl_hit,
                        "cluster_last": item["cluster_last"],
                        "cluster_pred_next": item["cluster_pred_next"],
                    }
                )
    else:
        min_total = backtest_folds * 24
        backtest_total = max(min_total, int(n_samples * 0.2))
        backtest_start = max(1, n_samples - backtest_total)
        block = max(1, (n_samples - backtest_start) // backtest_folds)

        for fold in range(backtest_folds):
            test_start = backtest_start + fold * block
            test_end = backtest_start + (fold + 1) * block if fold < backtest_folds - 1 else n_samples
            if test_start >= n_samples or test_end <= test_start:
                continue

            train_end = test_start
            val_size = max(24, int(train_end * 0.1))
            train_core_end = train_end - val_size
            if train_core_end < 120:
                continue

            train_idx = np.arange(0, train_core_end)
            val_idx = np.arange(train_core_end, train_end)
            test_idx = np.arange(test_start, test_end)

            model, _, _, scaler_x = run_single_fit(
                config=config,
                X_seq=X_seq,
                y_raw=y_raw,
                train_idx=train_idx,
                val_idx=val_idx,
                epochs=backtest_epochs,
                batch_size=batch_size,
                steps_per_execution=steps_per_execution,
                cache_dataset=cache_dataset,
                verbose=0,
            )

            n_features = X_seq.shape[-1]
            X_test = scaler_x.transform(X_seq[test_idx].reshape(-1, n_features)).reshape(len(test_idx), X_seq.shape[1], n_features).astype(np.float32)
            preds = predict_outputs_dict(model, X_test)

            for local_i, seq_i in enumerate(test_idx):
                pred_pack = {k: v[local_i : local_i + 1] for k, v in preds.items()}
                target_row = int(target_rows[seq_i])

                hist_labels = cluster_labels[:target_row]
                _, last_cluster, next_cluster = transition_from_labels(hist_labels, best_k) if len(hist_labels) > 1 else (
                    np.ones((best_k, best_k)) / best_k,
                    int(cluster_labels[max(0, target_row - 1)]),
                    int(cluster_labels[max(0, target_row - 1)]),
                )
                pred_win, pred_addl = combine_prediction(
                    pred=pred_pack,
                    win_cluster_prior=win_cluster_prior_matrix[target_row],
                    addl_cluster_prior=addl_cluster_prior_matrix[target_row],
                    repel_prior=repel_prior_matrix[target_row],
                    weights=blend_weights,
                )

                actual_win = df.loc[target_row, WIN_COLS].astype(int).tolist()
                actual_addl = int(df.loc[target_row, "Addl No."])
                win_hits, addl_hit = summarize_hits(pred_win, pred_addl, actual_win, actual_addl)

                records.append(
                    {
                        "fold": fold + 1,
                        "target_row": target_row,
                        "draw": int(df.loc[target_row, "Draw"]) if not pd.isna(df.loc[target_row, "Draw"]) else "",
                        "date": df.loc[target_row, "Date"].strftime("%Y-%m-%d") if not pd.isna(df.loc[target_row, "Date"]) else "",
                        "pred_win": " ".join(str(x) for x in pred_win),
                        "actual_win": " ".join(str(x) for x in sorted(actual_win)),
                        "pred_addl": pred_addl,
                        "actual_addl": actual_addl,
                        "win_hits": win_hits,
                        "addl_hit": addl_hit,
                        "cluster_last": last_cluster,
                        "cluster_pred_next": next_cluster,
                    }
                )

    if not records:
        empty_df = pd.DataFrame(
            columns=[
                "fold",
                "target_row",
                "draw",
                "date",
                "pred_win",
                "actual_win",
                "pred_addl",
                "actual_addl",
                "win_hits",
                "addl_hit",
                "cluster_last",
                "cluster_pred_next",
            ]
        )
        summary = {
            "samples": 0,
            "avg_win_hits": 0.0,
            "p_hit_ge_2": 0.0,
            "p_hit_ge_3": 0.0,
            "p_hit_ge_4": 0.0,
            "p_exact6": 0.0,
            "addl_acc": 0.0,
            "weighted_avg_win_hits": 0.0,
            "weighted_p_hit_ge_2": 0.0,
            "weighted_p_hit_ge_3": 0.0,
            "weighted_p_hit_ge_4": 0.0,
            "weighted_p_exact6": 0.0,
            "weighted_addl_acc": 0.0,
        }
        return empty_df, summary

    backtest_df = pd.DataFrame(records)
    n = len(backtest_df)
    recency_w = recency_weights_from_draw_date(
        backtest_df["draw"].values.astype(object),
        backtest_df["date"].values.astype(object),
        np.arange(n, dtype=np.float64),
    )
    weighted_m = hit_metrics_from_arrays(
        backtest_df["win_hits"].values.astype(np.float32),
        backtest_df["addl_hit"].values.astype(np.float32),
        sample_weights=recency_w,
    )
    weighted_p_exact6 = float(np.dot(recency_w, (backtest_df["win_hits"].values.astype(np.float32) == 6).astype(np.float32)) / (float(np.sum(recency_w)) + 1e-12))
    summary = {
        "samples": n,
        "avg_win_hits": float(backtest_df["win_hits"].mean()),
        "p_hit_ge_2": float((backtest_df["win_hits"] >= 2).mean()),
        "p_hit_ge_3": float((backtest_df["win_hits"] >= 3).mean()),
        "p_hit_ge_4": float((backtest_df["win_hits"] >= 4).mean()),
        "p_exact6": float((backtest_df["win_hits"] == 6).mean()),
        "addl_acc": float(backtest_df["addl_hit"].mean()),
        "weighted_avg_win_hits": float(weighted_m["avg_win_hits"]),
        "weighted_p_hit_ge_2": float(weighted_m["p_hit_ge2"]),
        "weighted_p_hit_ge_3": float(weighted_m["p_hit_ge3"]),
        "weighted_p_hit_ge_4": float(weighted_m["p_hit_ge4"]),
        "weighted_p_exact6": float(weighted_p_exact6),
        "weighted_addl_acc": float(weighted_m["addl_acc"]),
    }
    return backtest_df, summary


def html_table(df: pd.DataFrame, table_class: str = "table") -> str:
    if df.empty:
        return "<p><em>No rows.</em></p>"
    return df.to_html(index=False, classes=table_class, border=0, justify="center")


def build_single_html_report(
    output_path: Path,
    csv_path: Path,
    df_raw: pd.DataFrame,
    hw: Dict[str, object],
    cluster_info: Dict[str, object],
    best_config: ModelConfig,
    tuning_df: pd.DataFrame,
    final_metrics: Dict[str, float],
    backtest_df: pd.DataFrame,
    backtest_summary: Dict[str, float],
    prediction: Dict[str, object],
    diffusion_summary: Dict[str, object],
    transition_matrix: np.ndarray,
    dashboard: Dict[str, object],
    focus_last_n: int,
    feature_group_weights: Dict[str, float],
    feature_weight_tuning_df: pd.DataFrame,
    blend_component_df: pd.DataFrame,
    calibrated_blend: Dict[str, float],
    reward_summary: Dict[str, float],
) -> None:
    # Horizontal one-row metrics table so it consumes less vertical space.
    metrics_df = pd.DataFrame(
        [{k: float(v) for k, v in sorted(final_metrics.items())}]
    )
    transition_df = pd.DataFrame(
        transition_matrix,
        columns=[f"to_{i}" for i in range(transition_matrix.shape[1])],
    )
    transition_df.insert(0, "from_cluster", [f"from_{i}" for i in range(transition_matrix.shape[0])])

    tuning_show = tuning_df.copy()
    for c in [
        "dropout",
        "lr",
        "mean_val_loss",
        "mean_val_tune_obj",
        "mean_val_hit_score",
        "mean_val_avg_hits",
        "mean_val_p_hit_ge2",
        "mean_val_p_hit_ge3",
        "mean_val_p_hit_ge4",
        "mean_val_addl_acc",
    ]:
        if c in tuning_show.columns:
            tuning_show[c] = tuning_show[c].map(lambda x: f"{x:.6f}" if isinstance(x, float) else x)

    backtest_show = backtest_df.copy()
    if not backtest_show.empty:
        backtest_show["win_hits"] = backtest_show["win_hits"].astype(int)
        backtest_show["addl_hit"] = backtest_show["addl_hit"].astype(int)

    primary_grid_html = df_raw[["Draw", "Date"] + PRIMARY_COLS].to_html(index=False, classes="table grid", border=0)
    diffusion_trial_show = diffusion_summary["trial_df"].copy()
    for c in ["lr", "best_score", "best_hit_rate", "best_mse", "final_train_loss"]:
        if c in diffusion_trial_show.columns:
            diffusion_trial_show[c] = diffusion_trial_show[c].map(lambda x: f"{x:.6f}" if isinstance(x, float) else x)
    feature_weight_show = feature_weight_tuning_df.copy()
    if not feature_weight_show.empty:
        for c in feature_weight_show.columns:
            if c == "trial":
                continue
            if pd.api.types.is_numeric_dtype(feature_weight_show[c]):
                feature_weight_show[c] = feature_weight_show[c].map(lambda x: f"{x:.6f}" if isinstance(x, (float, np.floating)) else x)
    blend_component_show = blend_component_df.copy()
    if not blend_component_show.empty:
        for c in blend_component_show.columns:
            if c in {"component", "group"}:
                continue
            if pd.api.types.is_numeric_dtype(blend_component_show[c]):
                blend_component_show[c] = blend_component_show[c].map(lambda x: f"{x:.6f}" if isinstance(x, (float, np.floating)) else x)
    reward_show = pd.DataFrame([reward_summary]).copy()
    if not reward_show.empty:
        for c in reward_show.columns:
            if pd.api.types.is_numeric_dtype(reward_show[c]):
                reward_show[c] = reward_show[c].map(lambda x: f"{x:.6f}" if isinstance(x, (float, np.floating)) else x)
    if not metrics_df.empty:
        for c in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[c]):
                metrics_df[c] = metrics_df[c].map(lambda x: f"{x:.6f}" if isinstance(x, (float, np.floating)) else x)
    dashboard_json = json.dumps(dashboard, separators=(",", ":"))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ToTo Prediction Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg:#0b1220;
      --card:#131c2f;
      --card-2:#0f1728;
      --ink:#d7e2f0;
      --muted:#8da3be;
      --line:#1f2f4a;
      --accent:#34d399;
      --accent2:#60a5fa;
      --warn:#f59e0b;
    }}
    body {{
      margin: 0;
      padding: 18px;
      background: radial-gradient(circle at 10% 0%, #1a2843 0%, #0b1220 52%, #060b14 100%);
      font-family: "Segoe UI", "IBM Plex Sans", Tahoma, sans-serif;
      color: var(--ink);
    }}
    h1, h2, h3 {{
      margin: 10px 0;
      color: #e4ecf7;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      margin: 12px 0 16px 0;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
    }}
    .kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent2);
      border-radius: 9px;
      background: #0f1a2e;
      padding: 8px 10px;
      font-size: 13px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 12px;
    }}
    .chart-panel {{
      background: var(--card-2);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
    }}
    .plotly {{
      width: 100%;
      height: 360px;
    }}
    .long-rows {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    .long-row {{
      display: flex;
      gap: 12px;
      overflow-x: auto;
      padding-bottom: 8px;
    }}
    .long-chart {{
      flex: 0 0 640px;
      width: 640px;
      background: var(--card-2);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
    }}
    .long-scroll {{
      height: 460px;
      overflow-y: auto;
      overflow-x: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0b1322;
    }}
    .plotly-long {{
      width: 100%;
      min-height: 620px;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #0e1728;
      color: #d7e2f0;
    }}
    .table th, .table td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: center;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .table thead th {{
      background: #1f2a44;
      color: #e9f1fb;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .table tbody tr:nth-child(even) {{
      background: #0c1424;
    }}
    .grid-wrap {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 600;
      color: #8dd3ff;
      margin-bottom: 8px;
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <h1>ToTo Prediction</h1>
  <div class="sub">
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
    Source CSV: {csv_path} |
    Rows: {len(df_raw)}
  </div>

  <div class="card">
    <h2>Prediction Output</h2>
    <div class="kpi">
      <div class="pill"><strong>Predicted Win (Diffusion)</strong><br>{prediction["diffusion_win_numbers"]}</div>
      <div class="pill"><strong>Predicted Addl (Diffusion)</strong><br>{prediction["diffusion_addl_number"]}</div>
      <div class="pill"><strong>Predicted Win (Hybrid DL)</strong><br>{prediction["hybrid_win_numbers"]}</div>
      <div class="pill"><strong>Predicted Addl (Hybrid DL)</strong><br>{prediction["hybrid_addl_number"]}</div>
      <div class="pill"><strong>Cluster Last -> Next</strong><br>{prediction["cluster_last"]} -> {prediction["cluster_next"]}</div>
      <div class="pill"><strong>Predicted Sum / Mean</strong><br>{prediction["pred_sum"]} / {prediction["pred_mean"]:.2f}</div>
      <div class="pill"><strong>Low / High</strong><br>{prediction["low_count"]} / {prediction["high_count"]}</div>
      <div class="pill"><strong>Odd / Even</strong><br>{prediction["odd_count"]} / {prediction["even_count"]}</div>
    </div>
    <p class="small">
      Final set uses diffusion next-day extension; hybrid DL output is shown as reference.
    </p>
  </div>

  <div class="card">
    <h2>Hardware And Training Summary</h2>
    <div class="kpi">
      <div class="pill"><strong>GPU Count</strong><br>{hw["gpu_count"]}</div>
      <div class="pill"><strong>GPU Names</strong><br>{hw["gpu_names"]}</div>
      <div class="pill"><strong>Mixed Precision</strong><br>{hw["mixed_precision"]}</div>
      <div class="pill"><strong>XLA</strong><br>{hw["xla"]}</div>
      <div class="pill"><strong>Selected Clusters</strong><br>{cluster_info["best_k"]}</div>
      <div class="pill"><strong>Silhouette</strong><br>{cluster_info["silhouette"]:.4f}</div>
      <div class="pill"><strong>Best Seq Len</strong><br>{best_config.seq_len}</div>
      <div class="pill"><strong>Best LR</strong><br>{best_config.lr}</div>
    </div>
    <p class="small">Best config: {asdict(best_config)}</p>
  </div>

  <div class="card">
    <h2>Feature Group Weights</h2>
    <div class="kpi">
      <div class="pill"><strong>Primary</strong><br>{feature_group_weights["primary"]:.4f}</div>
      <div class="pill"><strong>Other</strong><br>{feature_group_weights["other"]:.4f}</div>
      <div class="pill"><strong>Repel</strong><br>{feature_group_weights["repel"]:.4f}</div>
      <div class="pill"><strong>Cluster Prior</strong><br>{feature_group_weights["cluster"]:.4f}</div>
      <div class="pill"><strong>Line Convergence</strong><br>{feature_group_weights["line"]:.4f}</div>
    </div>
    {"<h3>Feature Weight Tuning Trials</h3><div class='grid-wrap'>" + html_table(feature_weight_show) + "</div>" if not feature_weight_show.empty else "<p class='small'>Feature weight tuning was disabled for this run.</p>"}
  </div>

  <div class="card">
    <h2>Adaptive Blend Weights</h2>
    <div class="kpi">
      <div class="pill"><strong>model_soft</strong><br>{calibrated_blend["model_soft"]:.6f}</div>
      <div class="pill"><strong>model_set</strong><br>{calibrated_blend["model_set"]:.6f}</div>
      <div class="pill"><strong>cluster</strong><br>{calibrated_blend["cluster"]:.6f}</div>
      <div class="pill"><strong>repel</strong><br>{calibrated_blend["repel"]:.6f}</div>
      <div class="pill"><strong>addl_model</strong><br>{calibrated_blend["addl_model"]:.6f}</div>
      <div class="pill"><strong>addl_cluster</strong><br>{calibrated_blend["addl_cluster"]:.6f}</div>
    </div>
    {"<h3>Component Performance (weight basis)</h3><div class='grid-wrap'>" + html_table(blend_component_show) + "</div>" if not blend_component_show.empty else "<p class='small'>Adaptive component calibration was not available.</p>"}
  </div>

  <div class="card">
    <h2>Attention + Reward Self-Improvement</h2>
    <div class="kpi">
      <div class="pill"><strong>Reward Enabled</strong><br>{int(float(reward_summary.get("enabled", 0.0)))}</div>
      <div class="pill"><strong>Model Reverted</strong><br>{int(float(reward_summary.get("reverted", 0.0)))}</div>
      <div class="pill"><strong>Samples</strong><br>{int(float(reward_summary.get("samples", 0.0)))}</div>
      <div class="pill"><strong>Epochs</strong><br>{int(float(reward_summary.get("epochs", 0.0)))}</div>
      <div class="pill"><strong>Hit Score (Pre -> Post)</strong><br>{float(reward_summary.get("pre_hit_score", 0.0)):.4f} -> {float(reward_summary.get("post_hit_score", 0.0)):.4f}</div>
      <div class="pill"><strong>P(Hits >= 3) (Pre -> Post)</strong><br>{float(reward_summary.get("pre_p_hit_ge3", 0.0)):.4%} -> {float(reward_summary.get("post_p_hit_ge3", 0.0)):.4%}</div>
      <div class="pill"><strong>P(Hits >= 4) (Pre -> Post)</strong><br>{float(reward_summary.get("pre_p_hit_ge4", 0.0)):.4%} -> {float(reward_summary.get("post_p_hit_ge4", 0.0)):.4%}</div>
      <div class="pill"><strong>Avg Hits (Pre -> Post)</strong><br>{float(reward_summary.get("pre_avg_hits", 0.0)):.4f} -> {float(reward_summary.get("post_avg_hits", 0.0)):.4f}</div>
      <div class="pill"><strong>Mean / Max Reward</strong><br>{float(reward_summary.get("mean_reward", 0.0)):.4f} / {float(reward_summary.get("max_reward", 0.0)):.4f}</div>
      <div class="pill"><strong>Loss (Start -> End)</strong><br>{float(reward_summary.get("loss_start", 0.0)):.4f} -> {float(reward_summary.get("loss_end", 0.0)):.4f}</div>
    </div>
    {"<h3>Reward Detail</h3><div class='grid-wrap'>" + html_table(reward_show) + "</div>" if int(float(reward_summary.get('enabled', 0.0))) == 1 else "<p class='small'>Reward refinement skipped (insufficient samples or epochs).</p>"}
  </div>

  <div class="card">
    <h2>Diffusion Pattern Matching (DDPM)</h2>
    <div class="kpi">
      <div class="pill"><strong>Window Rows</strong><br>{diffusion_summary["window"]}</div>
      <div class="pill"><strong>Best Trial Score</strong><br>{diffusion_summary["trial_score"]:.4f}</div>
      <div class="pill"><strong>Pattern Hit Rate</strong><br>{diffusion_summary["trial_hit_rate"]:.4%}</div>
      <div class="pill"><strong>MSE</strong><br>{diffusion_summary["trial_mse"]:.6f}</div>
      <div class="pill"><strong>Future Samples</strong><br>{diffusion_summary["future_windows_used"]}</div>
      <div class="pill"><strong>Best DDPM Params</strong><br>{diffusion_summary["trial_best"]}</div>
    </div>
    <h3>Diffusion Tuning Trials</h3>
    <div class="grid-wrap">
      {html_table(diffusion_trial_show)}
    </div>
  </div>

  <div class="card">
    <h2>Interactive Charts</h2>
    <div class="chart-grid">
      <div class="chart-panel"><div id="cluster_scatter" class="plotly"></div></div>
      <div class="chart-panel"><div id="cluster_timeline" class="plotly"></div></div>
      <div class="chart-panel"><div id="cluster_profile" class="plotly"></div></div>
      <div class="chart-panel"><div id="transition_heat" class="plotly"></div></div>
      <div class="chart-panel"><div id="tuning_plot" class="plotly"></div></div>
      <div class="chart-panel"><div id="train_curve" class="plotly"></div></div>
      <div class="chart-panel"><div id="backtest_hist" class="plotly"></div></div>
      <div class="chart-panel"><div id="backtest_trend" class="plotly"></div></div>
      <div class="chart-panel"><div id="diff_next_prior" class="plotly"></div></div>
      <div class="chart-panel"><div id="diff_tune_curve" class="plotly"></div></div>
    </div>
    <h3>Long Grid Charts</h3>
    <div class="long-rows">
      <div class="long-row">
        <div class="long-chart"><div class="long-scroll"><div id="occurrence_heat" class="plotly-long"></div></div></div>
        <div class="long-chart"><div class="long-scroll"><div id="line_prior_heat" class="plotly-long"></div></div></div>
        <div class="long-chart"><div class="long-scroll"><div id="line_merge_heat" class="plotly-long"></div></div></div>
      </div>
      <div class="long-row">
        <div class="long-chart"><div class="long-scroll"><div id="diff_actual" class="plotly-long"></div></div></div>
        <div class="long-chart"><div class="long-scroll"><div id="diff_generated" class="plotly-long"></div></div></div>
        <div class="long-chart"><div class="long-scroll"><div id="diff_accuracy" class="plotly-long"></div></div></div>
        <div class="long-chart"><div class="long-scroll"><div id="diff_overlap" class="plotly-long"></div></div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Tuning And Validation</h2>
    <h3>Tuning Trials Table</h3>
    <div class="grid-wrap">
      {html_table(tuning_show)}
    </div>
    <h3>Final Test Metrics</h3>
    <div class="grid-wrap">
      {html_table(metrics_df)}
    </div>
  </div>

  <div class="card">
    <h2>{"Walk-Forward Backtest (Last " + str(focus_last_n) + " Draws)" if focus_last_n > 0 else "Walk-Forward Historical Backtest"}</h2>
    <div class="kpi">
      <div class="pill"><strong>Samples</strong><br>{backtest_summary["samples"]}</div>
      <div class="pill"><strong>Avg Win Hits</strong><br>{backtest_summary["avg_win_hits"]:.4f}</div>
      <div class="pill"><strong>P(Hits >= 2)</strong><br>{backtest_summary["p_hit_ge_2"]:.4%}</div>
      <div class="pill"><strong>P(Hits >= 3)</strong><br>{backtest_summary["p_hit_ge_3"]:.4%}</div>
      <div class="pill"><strong>P(Hits >= 4)</strong><br>{backtest_summary["p_hit_ge_4"]:.4%}</div>
      <div class="pill"><strong>P(Exact 6)</strong><br>{backtest_summary["p_exact6"]:.4%}</div>
      <div class="pill"><strong>Addl Accuracy</strong><br>{backtest_summary["addl_acc"]:.4%}</div>
      <div class="pill"><strong>W-Avg Win Hits</strong><br>{backtest_summary["weighted_avg_win_hits"]:.4f}</div>
      <div class="pill"><strong>W-P(Hits >= 3)</strong><br>{backtest_summary["weighted_p_hit_ge_3"]:.4%}</div>
      <div class="pill"><strong>W-P(Hits >= 4)</strong><br>{backtest_summary["weighted_p_hit_ge_4"]:.4%}</div>
      <div class="pill"><strong>W-Addl Accuracy</strong><br>{backtest_summary["weighted_addl_acc"]:.4%}</div>
    </div>
    <h3>Backtest Detail Table</h3>
    <div class="grid-wrap">
      {html_table(backtest_show)}
    </div>
  </div>

  <div class="card">
    <h2>Cluster Transition Matrix</h2>
    <div class="grid-wrap">
      {html_table(transition_df)}
    </div>
  </div>

  <div class="card">
    <h2>Primary Data Grid (Win_1..Win_6 + Addl No.)</h2>
    <details>
      <summary>Expand full primary dataset table ({len(df_raw)} rows)</summary>
      <div class="grid-wrap">
        {primary_grid_html}
      </div>
    </details>
  </div>

  <div class="card small">
    Lottery draws are random by design. This report quantifies pattern-learning behavior on historical data
    and walk-forward validation, but cannot guarantee future outcomes.
  </div>
  <script>
    const D = {dashboard_json};
    const baseLayout = {{
      paper_bgcolor: '#131c2f',
      plot_bgcolor: '#131c2f',
      font: {{color: '#d7e2f0', family: 'Segoe UI, Arial'}},
      margin: {{l: 52, r: 18, t: 42, b: 42}},
    }};

    const buildYTicks = (labels, maxTicks = 14) => {{
      const n = Array.isArray(labels) ? labels.length : 0;
      if (n === 0) return {{vals: [], text: []}};
      const step = Math.max(1, Math.floor(n / Math.max(1, maxTicks)));
      const vals = [];
      const text = [];
      for (let i = 0; i < n; i += step) {{
        vals.push(i);
        text.push(String(labels[i]));
      }}
      if (vals[vals.length - 1] !== n - 1) {{
        vals.push(n - 1);
        text.push(String(labels[n - 1]));
      }}
      return {{vals, text}};
    }};

    const squareHeatLayout = (title, xTitle, yTitle, yLabels, nCols = 49, maxHeight = 1800) => {{
      const nRows = Array.isArray(yLabels) ? yLabels.length : 1;
      const ticks = buildYTicks(yLabels, 14);
      const cell = 8;
      const height = Math.max(480, Math.min(maxHeight, 96 + cell * nRows));
      const width = Math.max(616, 140 + cell * nCols + 84);
      return {{
        ...baseLayout,
        title,
        height,
        width,
        xaxis: {{
          title: xTitle,
          dtick: 1,
          scaleanchor: 'y',
          scaleratio: 1,
          constrain: 'domain',
        }},
        yaxis: {{
          title: yTitle,
          tickmode: 'array',
          tickvals: ticks.vals,
          ticktext: ticks.text,
          autorange: 'reversed',
        }},
      }};
    }};

    Plotly.newPlot('cluster_scatter', [{{
      x: D.cluster.scatter_x,
      y: D.cluster.scatter_y,
      mode: 'markers',
      type: 'scattergl',
      marker: {{
        color: D.cluster.labels,
        colorscale: 'Turbo',
        size: 6,
        opacity: 0.85,
        showscale: true,
        colorbar: {{title: 'Cluster'}}
      }},
      text: D.cluster.draw_labels
    }}], {{...baseLayout, title: 'Cluster Shape Map (PCA)'}}, {{responsive:true}});

    Plotly.newPlot('cluster_timeline', [{{
      x: D.cluster.timeline_x,
      y: D.cluster.labels,
      mode: 'lines+markers',
      line: {{color: '#60a5fa', width: 1}},
      marker: {{size: 4, color: D.cluster.labels, colorscale: 'Turbo'}},
      type: 'scattergl'
    }}], {{...baseLayout, title: 'Cluster Timeline (Oldest -> Newest)', xaxis:{{title:'Index'}}, yaxis:{{title:'Cluster'}}}}, {{responsive:true}});

    Plotly.newPlot('cluster_profile', [{{
      z: D.cluster.profile_z,
      x: D.cluster.profile_x,
      y: D.cluster.profile_y,
      type: 'heatmap',
      colorscale: 'Viridis'
    }}], {{...baseLayout, title:'Cluster Pattern Heatmap (Win Columns Focus)'}}, {{responsive:true}});

    Plotly.newPlot('occurrence_heat', [{{
      z: D.occupancy.occ_z,
      x: D.occupancy.num_axis,
      y: D.occupancy.row_pos,
      type: 'heatmap',
      colorscale: 'Greens'
    }}], squareHeatLayout('Winning Number Occupancy Heatmap', 'Number (1-49)', 'Recent Draw Rows', D.occupancy.row_labels, D.occupancy.num_axis.length, 2100), {{responsive:true}});

    Plotly.newPlot('transition_heat', [{{
      z: D.cluster.transition_z,
      x: D.cluster.transition_x,
      y: D.cluster.transition_y,
      type: 'heatmap',
      colorscale: 'Magma'
    }}], {{...baseLayout, title:'Cluster Transition Probability Matrix'}}, {{responsive:true}});

    Plotly.newPlot('line_prior_heat', [{{
      z: D.line.prior_z,
      x: D.line.num_axis,
      y: D.line.row_pos,
      type: 'heatmap',
      colorscale: 'Cividis'
    }}], squareHeatLayout('Line Endpoint-Convergence Prior', 'Number', 'Recent Draw Rows', D.line.row_labels, D.line.num_axis.length, 2100), {{responsive:true}});

    Plotly.newPlot('line_merge_heat', [{{
      z: D.line.merge_z,
      x: D.line.num_axis,
      y: D.line.row_pos,
      type: 'heatmap',
      colorscale: 'YlOrBr'
    }}], squareHeatLayout('Line Endpoint Merge Density', 'Number', 'Recent Draw Rows', D.line.row_labels, D.line.num_axis.length, 2100), {{responsive:true}});

    Plotly.newPlot('tuning_plot', [
      {{
        x: D.tuning.trial,
        y: D.tuning.p_hit_ge2,
        mode: 'lines+markers',
        name: 'P(hit>=2)',
        line: {{color:'#f59e0b', width:2}}
      }},
      {{
        x: D.tuning.trial,
        y: D.tuning.p_hit_ge3,
        mode: 'lines+markers',
        name: 'P(hit>=3)',
        line: {{color:'#34d399', width:2}}
      }},
      {{
        x: D.tuning.trial,
        y: D.tuning.avg_hits,
        mode: 'lines+markers',
        name: 'Avg Win Hits',
        line: {{color:'#60a5fa', width:2}},
        yaxis: 'y2'
      }}
    ], {{
      ...baseLayout,
      title:'Tuning Objective Landscape',
      yaxis: {{title:'P(hit>=3)'}},
      yaxis2: {{title:'Avg Hits', overlaying:'y', side:'right'}},
      xaxis: {{title:'Trial'}}
    }}, {{responsive:true}});

    Plotly.newPlot('train_curve', [
      {{x: D.training.epoch, y: D.training.train_loss, mode:'lines', name:'train_loss', line:{{color:'#f59e0b'}}}},
      {{x: D.training.epoch, y: D.training.val_loss, mode:'lines', name:'val_loss', line:{{color:'#ef4444'}}}}
    ], {{...baseLayout, title:'Training Curve', xaxis:{{title:'Epoch'}}, yaxis:{{title:'Loss'}}}}, {{responsive:true}});

    Plotly.newPlot('backtest_hist', [{{
      x: D.backtest.win_hits,
      type: 'histogram',
      marker: {{color:'#22c55e'}},
      nbinsx: 8
    }}], {{...baseLayout, title:'Backtest Win-Hit Distribution{" (Last " + str(focus_last_n) + ")" if focus_last_n > 0 else ""}', xaxis:{{title:'Hit Count'}}, yaxis:{{title:'Frequency'}}}}, {{responsive:true}});

    Plotly.newPlot('backtest_trend', [
      {{
        x: D.backtest.index,
        y: D.backtest.win_hits,
        mode:'lines',
        name:'win_hits',
        line:{{color:'#60a5fa', width:1}}
      }},
      {{
        x: D.backtest.index,
        y: D.backtest.rolling,
        mode:'lines',
        name:'rolling_mean',
        line:{{color:'#f97316', width:2}}
      }}
    ], {{...baseLayout, title:'Backtest Hit Trend{" (Last " + str(focus_last_n) + ")" if focus_last_n > 0 else ""}', xaxis:{{title:'Sample'}}, yaxis:{{title:'Hit Count'}}}}, {{responsive:true}});

    Plotly.newPlot('diff_actual', [{{
      z: D.diffusion.actual_z,
      x: D.diffusion.num_axis,
      y: D.diffusion.row_pos,
      type: 'heatmap',
      colorscale: 'Turbo',
      zmin: 0,
      zmax: 1
    }}], squareHeatLayout('Actual Pattern Grid (Recent Window)', 'Number', 'Draw', D.diffusion.row_labels, D.diffusion.num_axis.length, 2400), {{responsive:true}});

    Plotly.newPlot('diff_generated', [{{
      z: D.diffusion.generated_z,
      x: D.diffusion.num_axis,
      y: D.diffusion.row_pos,
      type: 'heatmap',
      colorscale: 'Turbo',
      zmin: 0,
      zmax: 1
    }}], squareHeatLayout('Generated Pattern Grid (Diffusion)', 'Number', 'Draw', D.diffusion.row_labels, D.diffusion.num_axis.length, 2400), {{responsive:true}});

    Plotly.newPlot('diff_accuracy', [{{
      z: D.diffusion.accuracy_z,
      x: D.diffusion.num_axis,
      y: D.diffusion.row_pos,
      type: 'heatmap',
      colorscale: 'Viridis',
      zmin: 0,
      zmax: 1
    }}], squareHeatLayout('Accuracy Map (1 - abs(generated-actual))', 'Number', 'Draw', D.diffusion.row_labels, D.diffusion.num_axis.length, 2400), {{responsive:true}});

    Plotly.newPlot('diff_overlap', [{{
      z: D.diffusion.overlap_z,
      x: D.diffusion.num_axis,
      y: D.diffusion.row_pos,
      type: 'heatmap',
      colorscale: 'Greens',
      zmin: 0,
      zmax: 1
    }}], squareHeatLayout('Binary Overlap Map (Top-7 vs Actual)', 'Number', 'Draw', D.diffusion.row_labels, D.diffusion.num_axis.length, 2400), {{responsive:true}});

    Plotly.newPlot('diff_next_prior', [{{
      x: D.diffusion.num_axis,
      y: D.diffusion.next_prior,
      type: 'bar',
      marker: {{color:'#34d399'}}
    }}], {{...baseLayout, title:'Diffusion Next-Day Number Prior', xaxis:{{title:'Number'}}, yaxis:{{title:'Prior Score'}}}}, {{responsive:true}});

    Plotly.newPlot('diff_tune_curve', [{{
      x: D.diffusion.loss_epoch,
      y: D.diffusion.loss_curve,
      mode:'lines+markers',
      type:'scatter',
      marker: {{size:4, color:'#60a5fa'}},
      line: {{width:2, color:'#60a5fa'}}
    }}], {{...baseLayout, title:'Best Diffusion Trial Loss Curve', xaxis:{{title:'Epoch'}}, yaxis:{{title:'Train Loss'}}}}, {{responsive:true}});
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    if bool(args.hit_score_focused):
        prev_focus = int(args.focus_last_n)
        args.focus_last_n = 10 if int(args.focus_last_n) <= 0 else max(int(args.focus_last_n), 10)
        if bool(args.sweep_mode):
            args.tune_trials = max(int(args.tune_trials), 8)
            args.tune_epochs = max(int(args.tune_epochs), 8)
            args.multi_restarts = max(int(args.multi_restarts), 2)
            args.final_epochs = max(int(args.final_epochs), 42)
            args.backtest_folds = max(int(args.backtest_folds), 10)
            args.backtest_epochs = max(int(args.backtest_epochs), 12)
            args.reward_epochs = max(int(args.reward_epochs), 6)
            args.reward_window = max(int(args.reward_window), 220)
            args.blend_random_candidates = max(int(args.blend_random_candidates), 1500)
            args.blend_coordinate_iters = max(int(args.blend_coordinate_iters), 8)
            args.blend_coordinate_step = max(float(args.blend_coordinate_step), 0.34)
            args.blend_simplex_iters = max(int(args.blend_simplex_iters), 72)
            args.blend_simplex_step = max(float(args.blend_simplex_step), 0.20)
            args.blend_tail_iters = max(int(args.blend_tail_iters), 4)
            args.blend_tail_candidates = max(int(args.blend_tail_candidates), 160)
            args.backtest_local_random_candidates = max(int(args.backtest_local_random_candidates), 900)
            args.backtest_restarts = max(int(args.backtest_restarts), 1)
            args.restart_ensemble_topk = max(1, min(int(args.restart_ensemble_topk), int(args.backtest_restarts)))
            args.steps_per_execution = max(int(args.steps_per_execution), 24)
            args.dataset_cache = bool(args.dataset_cache) or bool(args.gpu_preallocate)
            args.diffusion_trials = max(1, int(args.diffusion_trials))
            args.diffusion_epochs = max(4, int(args.diffusion_epochs))
            args.diffusion_steps = max(28, int(args.diffusion_steps))
            args.diffusion_samples = max(2, int(args.diffusion_samples))
            args.diffusion_future_samples = max(48, int(args.diffusion_future_samples))
            args.diffusion_window = max(48, int(args.diffusion_window))
        else:
            args.tune_trials = max(int(args.tune_trials), 14)
            args.tune_epochs = max(int(args.tune_epochs), 10)
            args.multi_restarts = max(int(args.multi_restarts), 5)
            args.final_epochs = max(int(args.final_epochs), 64)
            args.backtest_folds = max(int(args.backtest_folds), 10)
            args.backtest_epochs = max(int(args.backtest_epochs), 20)
            args.reward_epochs = max(int(args.reward_epochs), 10)
            args.reward_window = max(int(args.reward_window), 320)
            args.blend_random_candidates = max(int(args.blend_random_candidates), 3600)
            args.blend_coordinate_iters = max(int(args.blend_coordinate_iters), 14)
            args.blend_coordinate_step = max(float(args.blend_coordinate_step), 0.42)
            args.blend_simplex_iters = max(int(args.blend_simplex_iters), 160)
            args.blend_simplex_step = max(float(args.blend_simplex_step), 0.24)
            args.blend_tail_iters = max(int(args.blend_tail_iters), 8)
            args.blend_tail_candidates = max(int(args.blend_tail_candidates), 360)
            args.backtest_local_random_candidates = max(int(args.backtest_local_random_candidates), 2600)
            args.backtest_restarts = max(int(args.backtest_restarts), 1)
            args.restart_ensemble_topk = max(1, min(int(args.restart_ensemble_topk), int(args.backtest_restarts)))
            args.steps_per_execution = max(int(args.steps_per_execution), 32)
            args.dataset_cache = bool(args.dataset_cache) or bool(args.gpu_preallocate)
            if str(args.perf_mode).lower() == "auto":
                args.perf_mode = "high"
            args.diffusion_trials = max(int(args.diffusion_trials), 5)
            args.diffusion_epochs = max(int(args.diffusion_epochs), 14)
            args.diffusion_steps = max(int(args.diffusion_steps), 72)
            args.diffusion_samples = max(int(args.diffusion_samples), 12)
            args.diffusion_future_samples = max(int(args.diffusion_future_samples), 240)
            args.diffusion_window = max(int(args.diffusion_window), 180)
        print(
            "[HIT-FOCUS] enabled: "
            f"focus_last_n {prev_focus}->{args.focus_last_n}, "
            f"seed={args.seed}, sweep_mode={'Y' if args.sweep_mode else 'N'}, "
            f"tune_trials={args.tune_trials}, tune_epochs={args.tune_epochs}, restarts={args.multi_restarts}, "
            f"blend_random_candidates={args.blend_random_candidates}, "
            f"coord_iters={args.blend_coordinate_iters}, coord_step={args.blend_coordinate_step:.2f}, "
            f"simplex_iters={args.blend_simplex_iters}, simplex_step={args.blend_simplex_step:.2f}, "
            f"tail_iters={args.blend_tail_iters}, tail_cands={args.blend_tail_candidates}, "
            f"bt_local_random={args.backtest_local_random_candidates}, bt_restarts={args.backtest_restarts}, "
            f"bt_topk={args.restart_ensemble_topk}, bt_opt_avg={'Y' if args.backtest_optimize_avg else 'N'}, "
            f"train_recent_w={'Y' if args.train_recency_weighted else 'N'}, "
            f"steps_per_exec={args.steps_per_execution}, cache={'Y' if args.dataset_cache else 'N'}, perf={args.perf_mode}, "
            f"diff_trials={args.diffusion_trials}, diff_epochs={args.diffusion_epochs}, diff_steps={args.diffusion_steps}"
        )
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_html:
        html_path = Path(args.output_html)
    else:
        html_path = Path("models") / f"ToTo_Tuned_OneFile_Report_{run_id}.html"

    print("=" * 90)
    print("ToTo Tuned Cluster DL + Single HTML Report")
    print("=" * 90)
    print(f"Seed: {args.seed}")
    print(f"Input CSV: {csv_path}")
    print(f"Output HTML: {html_path}")

    hw = configure_hardware(
        preallocate_gpu=bool(args.gpu_preallocate),
        gpu_batch_size_override=int(args.gpu_batch_size),
        perf_mode=str(args.perf_mode),
    )
    print(f"Hardware: GPUs={hw['gpu_count']} names={hw['gpu_names']} mixed_precision={hw['mixed_precision']} xla={hw['xla']}")

    df_raw = pd.read_csv(csv_path)
    df, primary_cols, other_cols = prepare_dataframe(df_raw)
    print(f"Loaded rows={len(df)}")

    diffusion_window = int(max(24, min(args.diffusion_window, len(df))))
    print(
        f"Running diffusion tuning: window={diffusion_window}, trials={args.diffusion_trials}, "
        f"epochs={args.diffusion_epochs}, steps={args.diffusion_steps}, samples/trial={args.diffusion_samples}"
    )
    diffusion_summary = run_diffusion_suite(
        df=df,
        window=diffusion_window,
        num_steps=max(20, int(args.diffusion_steps)),
        epochs=max(4, int(args.diffusion_epochs)),
        trials=max(1, int(args.diffusion_trials)),
        samples_per_trial=max(2, int(args.diffusion_samples)),
        future_samples=max(24, int(args.diffusion_future_samples)),
        gpu_batch_size=int(hw["batch_size"]),
        diffusion_batch_size=int(args.diffusion_batch_size),
    )
    print(
        f"Diffusion best score={diffusion_summary['trial_score']:.4f}, "
        f"hit_rate={diffusion_summary['trial_hit_rate']:.4f}, mse={diffusion_summary['trial_mse']:.5f}"
    )

    feature_group_weights = feature_group_weights_from_args(args)
    feature_weight_tuning_df = pd.DataFrame()
    if args.feature_weight_trials > 0:
        print(
            f"Running feature-weight tuning: trials={args.feature_weight_trials}, "
            f"epochs={args.feature_weight_epochs}"
        )
        feature_group_weights, feature_weight_tuning_df = tune_feature_group_weights(
            df=df,
            primary_cols=primary_cols,
            other_cols=other_cols,
            forced_clusters=args.clusters,
            base_weights=feature_group_weights,
            trials=args.feature_weight_trials,
            epochs=args.feature_weight_epochs,
            focus_last_n=args.focus_last_n,
            batch_size=int(hw["batch_size"]),
        )
        print(f"Selected feature group weights: {feature_group_weights}")

    pipe = build_feature_pipeline(
        df=df,
        primary_cols=primary_cols,
        other_cols=other_cols,
        group_weights=feature_group_weights,
        forced_clusters=args.clusters,
    )
    print(f"Feature group weights in use: {feature_group_weights}")

    df = pipe["df_features"]
    X_weighted = pipe["X_weighted"]
    cluster_labels = pipe["cluster_labels"]
    best_k = int(pipe["best_k"])
    silhouette = float(pipe["silhouette"])
    embedding = pipe["embedding"]
    cluster_profile_means = pipe["cluster_profile_means"]
    line_prior_matrix = pipe["line_prior_matrix"]
    line_merge_matrix = pipe["line_merge_matrix"]
    repel_prior_matrix = pipe["repel_prior_matrix"]
    win_cluster_prior_matrix = pipe["win_cluster_prior_matrix"]
    addl_cluster_prior_matrix = pipe["addl_cluster_prior_matrix"]
    cluster_transition_matrix = pipe["cluster_transition_matrix"]

    # Cache sequence datasets per seq_len for tuning.
    candidate_seq_lens = [18, 24, 30, 36]
    seq_cache: Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]] = {}
    for seq_len in candidate_seq_lens:
        seq_cache[seq_len] = build_sequences(X_weighted, df, seq_len, other_cols)
        print(f"Prepared sequences for seq_len={seq_len}: {seq_cache[seq_len][0].shape}")

    # Hyperparameter tuning via walk-forward validation.
    trial_configs = generate_trial_configs(args.tune_trials)
    recent_mode = args.focus_last_n > 0
    print(
        f"Running tuning: trials={len(trial_configs)}, folds={args.tune_folds}, "
        f"epochs={args.tune_epochs}, recent_focus={recent_mode}, last_n={args.focus_last_n}"
    )
    tuning_df = run_tuning(
        trial_configs=trial_configs,
        seq_cache=seq_cache,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=BLEND_WEIGHTS,
        tune_folds=args.tune_folds,
        tune_epochs=args.tune_epochs,
        focus_last_n=args.focus_last_n,
        batch_size=int(hw["batch_size"]),
        hit_score_focused=bool(args.hit_score_focused),
        steps_per_execution=int(args.steps_per_execution),
        cache_dataset=bool(args.dataset_cache),
        train_recency_weighted=bool(args.train_recency_weighted),
    )
    best_row = tuning_df.iloc[0]
    best_config = ModelConfig(
        seq_len=int(best_row["seq_len"]),
        lstm_units=int(best_row["lstm_units"]),
        gru_units=int(best_row["gru_units"]),
        dense_units=int(best_row["dense_units"]),
        dropout=float(best_row["dropout"]),
        lr=float(best_row["lr"]),
    )
    print(f"Best config selected: {asdict(best_config)}")

    # Final training using chronological train/val/test split.
    X_seq, y_raw, target_rows = seq_cache[best_config.seq_len]
    train_idx, val_idx, test_idx = split_train_val_test(len(X_seq))
    if args.focus_last_n > 0:
        restart_eval_idx = recent_seq_indices(len(X_seq), args.focus_last_n, min_train=max(120, best_config.seq_len * 4))
        if len(restart_eval_idx) == 0:
            restart_eval_idx = val_idx
        restart_eval_weights = recency_weights_for_target_rows(df, target_rows[np.asarray(restart_eval_idx, dtype=np.int32)])
        if bool(args.train_recency_weighted):
            final_train_sw = sharpen_recency_weights(
                recency_weights_for_target_rows(df, target_rows[np.asarray(train_idx, dtype=np.int32)]),
                power=1.35,
                floor=0.32,
            )
            final_val_sw = recency_weights_for_target_rows(df, target_rows[np.asarray(val_idx, dtype=np.int32)])
        else:
            final_train_sw = None
            final_val_sw = None
    else:
        restart_eval_idx = val_idx
        restart_eval_weights = None
        final_train_sw = None
        final_val_sw = None

    n_restarts = max(1, int(args.multi_restarts))
    best_restart_score = -1e18
    best_restart_seed = int(args.seed)
    best_restart_metrics: Dict[str, float] = {
        "avg_win_hits": 0.0,
        "p_hit_ge2": 0.0,
        "p_hit_ge3": 0.0,
        "p_hit_ge4": 0.0,
        "addl_acc": 0.0,
        "hit_score": 0.0,
    }
    best_restart_val_loss = float("inf")
    best_weights: List[np.ndarray] | None = None
    best_scaler_x: StandardScaler | None = None
    best_history_dict: Dict[str, List[float]] = {}

    for restart_i in range(n_restarts):
        restart_seed = int(args.seed) + 97 * restart_i
        set_global_seed(restart_seed)
        model_i, history_i, _, scaler_i = run_single_fit(
            config=best_config,
            X_seq=X_seq,
            y_raw=y_raw,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=args.final_epochs,
            batch_size=int(hw["batch_size"]),
            steps_per_execution=int(args.steps_per_execution),
            cache_dataset=bool(args.dataset_cache),
            train_sample_weights=final_train_sw,
            val_sample_weights=final_val_sw,
            verbose=0,
        )
        restart_metrics = evaluate_hit_score(
            model=model_i,
            scaler_x=scaler_i,
            X_seq=X_seq,
            eval_idx=restart_eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=BLEND_WEIGHTS,
            sample_weights=restart_eval_weights,
        )
        restart_score = float(blend_objective(restart_metrics, focused=bool(args.hit_score_focused)))
        restart_val_loss = float(np.min(history_i.history.get("val_loss", [math.inf])))
        print(
            f"[RESTART] {restart_i + 1}/{n_restarts} seed={restart_seed} "
            f"score={restart_score:.4f} p>=3={restart_metrics['p_hit_ge3']:.4f} "
            f"p>=4={restart_metrics['p_hit_ge4']:.4f} avg={restart_metrics['avg_win_hits']:.4f} "
            f"val_loss={restart_val_loss:.4f}"
        )
        if (restart_score > best_restart_score + 1e-10) or (
            abs(restart_score - best_restart_score) <= 1e-10 and restart_val_loss < best_restart_val_loss
        ):
            best_restart_score = restart_score
            best_restart_seed = restart_seed
            best_restart_metrics = dict(restart_metrics)
            best_restart_val_loss = restart_val_loss
            best_weights = model_i.get_weights()
            best_scaler_x = scaler_i
            best_history_dict = {k: [float(x) for x in v] for k, v in history_i.history.items()}

    if best_weights is None or best_scaler_x is None:
        raise RuntimeError("Final training restarts failed to produce a model.")

    tf.keras.backend.clear_session()
    model = build_model(
        best_config,
        n_features=X_seq.shape[-1],
        n_aux=y_raw["aux"].shape[-1],
        steps_per_execution=int(args.steps_per_execution),
    )
    model.set_weights(best_weights)
    scaler_x = best_scaler_x
    final_history = keras.callbacks.History()
    final_history.history = best_history_dict
    print(
        "Final training done. "
        f"restarts={n_restarts} selected_seed={best_restart_seed} "
        f"restart_score={best_restart_score:.4f} "
        f"p>=3={best_restart_metrics['p_hit_ge3']:.4f} p>=4={best_restart_metrics['p_hit_ge4']:.4f} "
        f"epochs={len(final_history.history.get('loss', []))} "
        f"best_val_loss={best_restart_val_loss:.4f}"
    )

    aux_scaler = StandardScaler()
    aux_scaler.fit(y_raw["aux"][train_idx])

    reward_summary = reward_guided_refinement(
        model=model,
        scaler_x=scaler_x,
        aux_scaler=aux_scaler,
        X_seq=X_seq,
        y_raw=y_raw,
        target_rows=target_rows,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=BLEND_WEIGHTS,
        reward_window=int(args.reward_window),
        min_samples=int(args.reward_min_samples),
        epochs=int(args.reward_epochs),
        batch_size=int(hw["batch_size"]),
    )
    if int(float(reward_summary.get("enabled", 0.0))) == 1:
        reverted_msg = " (reverted to pre-reward weights)" if int(float(reward_summary.get("reverted", 0.0))) == 1 else ""
        print(
            "Reward refinement done. "
            f"samples={int(reward_summary['samples'])} "
            f"epochs={int(reward_summary['epochs'])} "
            f"hit_score_delta={reward_summary['delta_hit_score']:.4f}"
            f"{reverted_msg}"
        )
    else:
        print("Reward refinement skipped (insufficient samples or epochs).")

    # Evaluate on holdout test.
    n_features = X_seq.shape[-1]
    X_test = scaler_x.transform(X_seq[test_idx].reshape(-1, n_features)).reshape(len(test_idx), X_seq.shape[1], n_features).astype(np.float32)
    y_test: Dict[str, np.ndarray] = {}
    for key, arr in y_raw.items():
        if key == "aux":
            y_test[key] = aux_scaler.transform(arr[test_idx]).astype(np.float32)
        elif key == "win_set":
            y_test[key] = arr[test_idx].astype(np.float32)
        else:
            y_test[key] = arr[test_idx].astype(np.int32)
    test_ds = make_dataset(
        X_test,
        y_test,
        batch_size=int(hw["batch_size"]),
        shuffle=False,
        cache=bool(args.dataset_cache),
    )
    final_test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    # Blend-weight calibration and selection on validation slice.
    selected_blend = dict(BLEND_WEIGHTS)
    selected_blend_metrics: Dict[str, float] = {
        "avg_win_hits": -1.0,
        "p_hit_ge2": -1.0,
        "p_hit_ge3": -1.0,
        "p_hit_ge4": -1.0,
        "addl_acc": -1.0,
        "hit_score": -1.0,
    }
    if args.focus_last_n > 0:
        blend_eval_idx = recent_seq_indices(len(X_seq), args.focus_last_n, min_train=max(120, best_config.seq_len * 4))
        if len(blend_eval_idx) == 0:
            blend_eval_idx = val_idx
        blend_eval_weights = recency_weights_for_target_rows(df, target_rows[np.asarray(blend_eval_idx, dtype=np.int32)])
    else:
        blend_eval_idx = val_idx
        blend_eval_weights = None
    blend_eval_preds: Dict[str, np.ndarray] | None = None
    if len(blend_eval_idx) > 0:
        n_features = X_seq.shape[-1]
        X_blend_eval = scaler_x.transform(X_seq[blend_eval_idx].reshape(-1, n_features)).reshape(
            len(blend_eval_idx), X_seq.shape[1], n_features
        ).astype(np.float32)
        blend_eval_preds = predict_outputs_dict(model, X_blend_eval)

    calibrated_blend, blend_component_df = calibrate_blend_weights_by_component_performance(
        model=model,
        scaler_x=scaler_x,
        X_seq=X_seq,
        eval_idx=blend_eval_idx,
        target_rows=target_rows,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        base_weights=BLEND_WEIGHTS,
        sample_weights=blend_eval_weights,
        cached_preds=blend_eval_preds,
    )
    print(f"Adaptive calibrated blend seed: {calibrated_blend}")
    if not blend_component_df.empty:
        print("Component performance used for weighting:")
        print(blend_component_df.to_string(index=False))

    blend_candidates = [dict(calibrated_blend)]
    blend_candidates.extend(generate_backtest_blend_candidates(calibrated_blend, num_random=int(args.blend_random_candidates)))
    # Keep a few hand-crafted variants as fallback.
    blend_candidates.extend(generate_blend_candidates(calibrated_blend))
    best_blend_score = -1e9
    blend_focused = bool(args.hit_score_focused)
    for cand in blend_candidates:
        val_hit_metrics = evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=blend_eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=cand,
            sample_weights=blend_eval_weights,
            cached_preds=blend_eval_preds,
        )
        blend_score = blend_objective(val_hit_metrics, focused=blend_focused)
        if blend_score > best_blend_score:
            best_blend_score = blend_score
            selected_blend = dict(cand)
            selected_blend_metrics = val_hit_metrics

    def eval_blend_metrics(cand: Dict[str, float]) -> Dict[str, float]:
        return evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=blend_eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=cand,
            sample_weights=blend_eval_weights,
            cached_preds=blend_eval_preds,
        )

    selected_blend, selected_blend_metrics, best_blend_score = coordinate_ascent_blend_search(
        start_weights=selected_blend,
        eval_fn=eval_blend_metrics,
        focused=blend_focused,
        max_iters=int(args.blend_coordinate_iters),
        init_step=float(args.blend_coordinate_step),
    )
    selected_blend, selected_blend_metrics, best_blend_score = nelder_mead_blend_search(
        start_weights=selected_blend,
        eval_fn=eval_blend_metrics,
        focused=blend_focused,
        max_iters=int(args.blend_simplex_iters),
        step=float(args.blend_simplex_step),
    )
    if blend_focused:
        selected_blend, selected_blend_metrics, best_blend_score = tail_focus_blend_refine(
            start_weights=selected_blend,
            eval_fn=eval_blend_metrics,
            max_iters=int(args.blend_tail_iters),
            candidates_per_iter=int(args.blend_tail_candidates),
        )

    blend_scope = f"last {args.focus_last_n}" if args.focus_last_n > 0 else "validation"
    print(f"Selected blend ({blend_scope}-optimized): {selected_blend}")
    print(f"Selected blend metrics on {blend_scope}: {selected_blend_metrics}")
    print(f"Selected blend objective on {blend_scope}: {best_blend_score:.6f}")

    # Walk-forward expanding-window backtest for historical validation.
    if args.focus_last_n > 0:
        print(f"Running strict walk-forward backtest on last {args.focus_last_n} draws, epochs={args.backtest_epochs}")
    else:
        print(f"Running walk-forward backtest: folds={args.backtest_folds}, epochs={args.backtest_epochs}")
    backtest_df, backtest_summary = run_backtest(
        df=df,
        cluster_labels=cluster_labels,
        best_k=best_k,
        config=best_config,
        X_seq=X_seq,
        y_raw=y_raw,
        target_rows=target_rows,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=selected_blend,
        backtest_folds=args.backtest_folds,
        backtest_epochs=args.backtest_epochs,
        focus_last_n=args.focus_last_n,
        batch_size=int(hw["batch_size"]),
        local_random_candidates=int(args.backtest_local_random_candidates),
        optimize_avg=bool(args.backtest_optimize_avg),
        backtest_restarts=int(args.backtest_restarts),
        restart_ensemble_topk=int(args.restart_ensemble_topk),
        steps_per_execution=int(args.steps_per_execution),
        cache_dataset=bool(args.dataset_cache),
        train_recency_weighted=bool(args.train_recency_weighted),
    )

    # Final prediction for next draw using latest sequence + cluster trend priors.
    transition_matrix = cluster_transition_matrix
    cluster_last = int(cluster_labels[-1])
    cluster_next = int(np.argmax(transition_matrix[cluster_last]))

    next_cluster_mask = cluster_labels == cluster_next
    if np.any(next_cluster_mask):
        win_counts_next = np.bincount(df.loc[next_cluster_mask, WIN_COLS].values.astype(int).reshape(-1), minlength=50)[1:]
        addl_counts_next = np.bincount(df.loc[next_cluster_mask, "Addl No."].values.astype(int), minlength=50)[1:]
        win_cluster_next_prior = win_counts_next / max(1.0, float(win_counts_next.sum()))
        addl_cluster_next_prior = addl_counts_next / max(1.0, float(addl_counts_next.sum()))
    else:
        win_cluster_next_prior = np.ones(49, dtype=np.float32) / 49.0
        addl_cluster_next_prior = np.ones(49, dtype=np.float32) / 49.0

    # Build one-step-ahead repel prior (for row n, based on rows [0..n-1]).
    dummy_next = df.iloc[[-1]].copy()
    next_df = pd.concat([df, dummy_next], ignore_index=True)
    repel_next_prior = build_repel_system(next_df)["repel_prior"][-1]

    latest_seq = scaler_x.transform(X_seq[-1].reshape(-1, n_features)).reshape(1, X_seq.shape[1], n_features).astype(np.float32)
    pred_latest = predict_outputs_dict(model, latest_seq)
    pred_win, pred_addl = combine_prediction(
        pred=pred_latest,
        win_cluster_prior=win_cluster_next_prior,
        addl_cluster_prior=addl_cluster_next_prior,
        repel_prior=repel_next_prior,
        weights=selected_blend,
    )

    diff_pred_win = [int(x) for x in diffusion_summary["diffusion_pred_win"]]
    diff_pred_addl = int(diffusion_summary["diffusion_pred_addl"])
    pred_stats = {
        "diffusion_win_numbers": diff_pred_win,
        "diffusion_addl_number": diff_pred_addl,
        "hybrid_win_numbers": pred_win,
        "hybrid_addl_number": pred_addl,
        "cluster_last": int(cluster_last),
        "cluster_next": int(cluster_next),
        "pred_sum": int(sum(diff_pred_win)),
        "pred_mean": float(np.mean(diff_pred_win)),
        "low_count": int(sum(1 for n in diff_pred_win if n <= 24)),
        "high_count": int(sum(1 for n in diff_pred_win if n > 24)),
        "odd_count": int(sum(1 for n in diff_pred_win if n % 2 == 1)),
        "even_count": int(sum(1 for n in diff_pred_win if n % 2 == 0)),
    }

    # Interactive dashboard payload (Grafana-style Plotly charts).
    history_loss = [float(x) for x in final_history.history.get("loss", [])]
    history_val_loss = [float(x) for x in final_history.history.get("val_loss", [])]
    history_epoch = list(range(1, len(history_loss) + 1))

    recent_window = min(180, len(df))
    recent_idx = np.arange(len(df) - recent_window, len(df))
    occ_heat = np.zeros((recent_window, 49), dtype=np.float32)
    for i, row_i in enumerate(recent_idx):
        wins = df.loc[row_i, WIN_COLS].astype(int).values
        addl = int(df.loc[row_i, "Addl No."])
        occ_heat[i, wins - 1] = 1.0
        occ_heat[i, addl - 1] = max(occ_heat[i, addl - 1], 0.6)

    row_labels = [str(int(df.loc[i, "Draw"])) if not pd.isna(df.loc[i, "Draw"]) else str(i) for i in recent_idx]
    row_pos = list(range(recent_window))

    if backtest_df.empty:
        backtest_hits = []
        backtest_rolling = []
        backtest_idx = []
    else:
        backtest_hits = backtest_df["win_hits"].astype(float).tolist()
        rolling_win = 5 if args.focus_last_n > 0 else 20
        backtest_rolling = backtest_df["win_hits"].rolling(rolling_win, min_periods=1).mean().astype(float).tolist()
        backtest_idx = list(range(len(backtest_df)))

    dashboard = {
        "cluster": {
            "scatter_x": embedding[:, 0].astype(float).tolist(),
            "scatter_y": embedding[:, 1].astype(float).tolist(),
            "labels": cluster_labels.astype(int).tolist(),
            "draw_labels": [
                str(int(v)) if not pd.isna(v) else str(i)
                for i, v in enumerate(df["Draw"].tolist())
            ],
            "timeline_x": list(range(len(cluster_labels))),
            "profile_z": cluster_profile_means.values.astype(float).tolist(),
            "profile_x": cluster_profile_means.columns.tolist(),
            "profile_y": [str(x) for x in cluster_profile_means.index.tolist()],
            "transition_z": transition_matrix.astype(float).tolist(),
            "transition_x": [f"to_{i}" for i in range(transition_matrix.shape[1])],
            "transition_y": [f"from_{i}" for i in range(transition_matrix.shape[0])],
        },
        "occupancy": {
            "occ_z": occ_heat.astype(float).tolist(),
            "num_axis": list(range(1, 50)),
            "row_labels": row_labels,
            "row_pos": row_pos,
        },
        "line": {
            "prior_z": line_prior_matrix[recent_idx].astype(float).tolist(),
            "merge_z": line_merge_matrix[recent_idx].astype(float).tolist(),
            "num_axis": list(range(1, 50)),
            "row_labels": row_labels,
            "row_pos": row_pos,
        },
        "tuning": {
            "trial": tuning_df["trial"].astype(int).tolist(),
            "p_hit_ge2": tuning_df["mean_val_p_hit_ge2"].astype(float).tolist(),
            "p_hit_ge3": tuning_df["mean_val_p_hit_ge3"].astype(float).tolist(),
            "avg_hits": tuning_df["mean_val_avg_hits"].astype(float).tolist(),
            "loss": tuning_df["mean_val_loss"].astype(float).tolist(),
        },
        "training": {
            "epoch": history_epoch,
            "train_loss": history_loss,
            "val_loss": history_val_loss,
        },
        "backtest": {
            "index": backtest_idx,
            "win_hits": backtest_hits,
            "rolling": backtest_rolling,
        },
        "diffusion": {
            "actual_z": diffusion_summary["target_role"].astype(float).tolist(),
            "generated_z": diffusion_summary["generated_role"].astype(float).tolist(),
            "accuracy_z": diffusion_summary["accuracy_map"].astype(float).tolist(),
            "overlap_z": diffusion_summary["overlap_map"].astype(float).tolist(),
            "row_labels": diffusion_summary["row_labels"],
            "row_pos": list(range(len(diffusion_summary["row_labels"]))),
            "num_axis": list(range(1, 50)),
            "next_prior": diffusion_summary["diffusion_next_row_prior"].astype(float).tolist(),
            "loss_epoch": list(range(1, len(diffusion_summary["trial_loss_curve"]) + 1)),
            "loss_curve": [float(x) for x in diffusion_summary["trial_loss_curve"]],
        },
    }

    cluster_info = {"best_k": best_k, "silhouette": silhouette}
    build_single_html_report(
        output_path=html_path,
        csv_path=csv_path,
        df_raw=df_raw,
        hw=hw,
        cluster_info=cluster_info,
        best_config=best_config,
        tuning_df=tuning_df,
        final_metrics=final_test_metrics,
        backtest_df=backtest_df,
        backtest_summary=backtest_summary,
        prediction=pred_stats,
        diffusion_summary=diffusion_summary,
        transition_matrix=transition_matrix,
        dashboard=dashboard,
        focus_last_n=args.focus_last_n,
        feature_group_weights=feature_group_weights,
        feature_weight_tuning_df=feature_weight_tuning_df,
        blend_component_df=blend_component_df,
        calibrated_blend=calibrated_blend,
        reward_summary=reward_summary,
    )

    print("")
    print("Prediction (latest):")
    print(f"Diffusion Win: {diff_pred_win}")
    print(f"Diffusion Addl: {diff_pred_addl}")
    print(f"Hybrid Win: {pred_win}")
    print(f"Hybrid Addl: {pred_addl}")
    print("")
    print("Backtest summary:")
    print(backtest_summary)
    print("")
    print(f"Single HTML report generated: {html_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
