"""
ToTo deep learning with:
1) automated hyperparameter tuning (walk-forward validation),
2) expanding-window historical backtest,
3) one single self-contained HTML report (all visuals + tables + metrics).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
    "model_soft": 1.1820,
    "graph": 1.2060,
    "embed": 0.8120,
    "cluster": 0.8900,
    "repel": 0.1240,
    "addl_model": 0.7784,
}

WIN_BLEND_KEYS = ["model_soft", "graph", "embed", "cluster"]
ADDL_BLEND_KEYS = ["addl_model"]

DEFAULT_FEATURE_GROUP_WEIGHTS = {
    "primary": 4.0,
    "other": 1.25,
    "repel": 1.20,
    "cluster": 1.35,
    "graph": 1.60,
    "embed": 1.45,
}


@dataclass
class ModelConfig:
    seq_len: int
    lstm_units: int
    gru_units: int
    dense_units: int
    dropout: float
    lr: float


@dataclass
class RewardReranker:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coef: np.ndarray
    intercept: float
    reward_scale: float = 0.55

    def score(self, feat: np.ndarray) -> float:
        x = np.asarray(feat, dtype=np.float64).reshape(-1)
        z = (x - self.feature_mean) / self.feature_std
        return float(self.intercept + np.dot(z, self.coef))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ToTo Predication Report"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Global random seed")
    parser.add_argument("--sweep-mode", action="store_true", help="Use lighter settings for automated multi-run sweeps")
    parser.add_argument("--csv", default="ToTo-12_Mar_2026.csv", help="Input CSV file path")
    parser.add_argument("--clusters", type=int, default=0, help="Force KMeans clusters (0=auto)")
    parser.add_argument("--tune-trials", type=int, default=14, help="Random trials sampled from config space")
    parser.add_argument("--tune-folds", type=int, default=3, help="Walk-forward folds per tuning trial")
    parser.add_argument("--tune-epochs", type=int, default=10, help="Epoch cap for each tuning fold")
    parser.add_argument(
        "--tune-multifidelity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use two-stage multi-fidelity tuning (short stage + full stage on shortlist)",
    )
    parser.add_argument("--tune-mf-stage1-epochs", type=int, default=5, help="Epoch cap for stage-1 multi-fidelity tuning")
    parser.add_argument("--tune-mf-keep-ratio", type=float, default=0.45, help="Top ratio kept from stage-1 into full stage-2 tuning")
    parser.add_argument("--multi-restarts", type=int, default=5, help="Number of final-training restarts (best restart kept by hit objective)")
    parser.add_argument("--final-epochs", type=int, default=64, help="Final training epoch cap")
    parser.add_argument("--backtest-folds", type=int, default=10, help="Expanding-window backtest folds")
    parser.add_argument("--backtest-epochs", type=int, default=20, help="Epoch cap per backtest fold")
    parser.add_argument(
        "--focus-last-n",
        type=int,
        default=24,
        help="Optimize and backtest only the most recent N draws (0 disables recent-focus mode)",
    )
    parser.add_argument("--output-html", default="", help="Output HTML path (optional)")
    parser.add_argument("--w-primary", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["primary"], help="Feature weight for Win_1..Win_6 + Addl No.")
    parser.add_argument("--w-other", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["other"], help="Feature weight for non-primary scalar columns")
    parser.add_argument("--w-repel", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["repel"], help="Feature weight for repel features")
    parser.add_argument("--w-cluster", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["cluster"], help="Feature weight for cluster-prior features")
    parser.add_argument("--w-graph", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["graph"], help="Feature weight for graph co-occurrence features")
    parser.add_argument("--w-embed", type=float, default=DEFAULT_FEATURE_GROUP_WEIGHTS["embed"], help="Feature weight for embedding-pattern features")
    parser.add_argument("--w-line", dest="w_graph", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
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
        "--backtest-no-retrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the final trained model for backtest inference (skip per-step retraining for speed)",
    )
    parser.add_argument(
        "--backtest-optimize-avg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize strict walk-forward local blend for avg_win_hits emphasis",
    )
    parser.add_argument(
        "--latest-priority-n",
        type=int,
        default=5,
        help="Latest N backtest samples to strongly prioritize in optimization objectives (0 disables tail-priority)",
    )
    parser.add_argument(
        "--latest-priority-boost",
        type=float,
        default=3.4,
        help="Multiplicative recency boost applied to the latest-priority window",
    )
    parser.add_argument(
        "--latest-priority-lambda",
        type=float,
        default=2.2,
        help="Extra objective weight on latest-priority win-hit performance during backtest blend search",
    )
    parser.add_argument(
        "--latest-priority-addl-lambda",
        type=float,
        default=1.6,
        help="Extra objective weight on latest-priority additional-number hit performance",
    )
    parser.add_argument(
        "--uncertainty-dynamic-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply entropy/temperature-calibrated dynamic per-draw blend weighting",
    )
    parser.add_argument("--candidate-max-pool", type=int, default=48, help="Maximum unique win-set candidates scored per prediction")
    parser.add_argument("--candidate-gumbel-samples", type=int, default=28, help="Stochastic Gumbel-top-k candidate samples per prediction")
    parser.add_argument("--candidate-diversity-lambda", type=float, default=0.11, help="Penalty strength for low-spread candidate sets")
    parser.add_argument(
        "--expected-hit-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add expected-hit utility term during candidate ranking (avg-hit oriented)",
    )
    parser.add_argument("--expected-hit-lambda", type=float, default=2.20, help="Weight for expected-hit utility term")
    parser.add_argument("--expected-hit-synergy-lambda", type=float, default=0.48, help="Weight for component-consensus utility term")
    parser.add_argument("--anti-repeat-window", type=int, default=10, help="Recent prediction window used to penalize repeated win sets")
    parser.add_argument("--anti-repeat-lambda", type=float, default=1.25, help="Penalty strength against repeated/high-overlap predicted win sets")
    parser.add_argument(
        "--avg-hit-focused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profile that prioritizes avg_win_hits over tail-only metrics",
    )
    parser.add_argument(
        "--reward-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reward-model reranking on candidate win sets",
    )
    parser.add_argument("--reward-window", type=int, default=72, help="Recent tuning rows used to fit reward reranker")
    parser.add_argument("--reward-a", type=float, default=1.0, help="Reward coefficient for hit count")
    parser.add_argument("--reward-b", type=float, default=2.4, help="Reward bonus for hits>=3")
    parser.add_argument("--reward-c", type=float, default=6.8, help="Reward bonus for hits>=4")
    parser.add_argument("--reward-hardneg-boost", type=float, default=0.9, help="Extra weight for hard negatives (hits 1-2) in reward fitting")
    parser.add_argument("--reward-ridge", type=float, default=0.18, help="L2 regularization for reward reranker fit")
    parser.add_argument("--reward-scale", type=float, default=0.55, help="Blend scale for reward reranker score at inference time")
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


def feature_group_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "primary": float(args.w_primary),
        "other": float(args.w_other),
        "repel": float(args.w_repel),
        "cluster": float(args.w_cluster),
        "graph": float(args.w_graph),
        "embed": float(args.w_embed),
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

    cooc_system = build_cooccurrence_system(df)
    cooc_prior_matrix = cooc_system["cooc_prior"]
    embed_system = build_embedding_pattern_system(df)
    embed_prior_matrix = np.asarray(embed_system["embed_prior"], dtype=np.float32)
    addl_embed_prior_matrix = np.asarray(embed_system["addl_embed_prior"], dtype=np.float32)
    repel_system = build_repel_system(df)
    repel_prior_matrix = repel_system["repel_prior"]

    win_cluster_prior_matrix, addl_cluster_prior_matrix, cluster_transition_matrix = build_dynamic_cluster_priors(
        df=df,
        cluster_labels=cluster_labels,
        k=best_k,
    )

    cooc_cols = [f"Cooc_{i+1}" for i in range(49)]
    embed_cols = [f"EmbedPrior_{i+1}" for i in range(49)]
    repel_cols = [f"Repel_{i+1}" for i in range(49)]
    cluster_prior_cols = [f"ClusterPrior_{i+1}" for i in range(49)]

    extra_features = pd.DataFrame(
        np.hstack(
            [
                cooc_prior_matrix,
                embed_prior_matrix,
                repel_prior_matrix,
                win_cluster_prior_matrix,
            ]
        ),
        columns=cooc_cols + embed_cols + repel_cols + cluster_prior_cols,
        index=df.index,
    )
    df_features = pd.concat([df.copy(), extra_features], axis=1)

    feature_cols = (
        base_feature_cols
        + cooc_cols
        + embed_cols
        + repel_cols
        + cluster_prior_cols
    )
    feature_weights = np.array(
        [group_weights["primary"]] * len(primary_cols)
        + [group_weights["other"]] * len(other_cols)
        + [group_weights["graph"]] * len(cooc_cols)
        + [group_weights["embed"]] * len(embed_cols)
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
        "graph_prior_matrix": cooc_prior_matrix,
        "embed_prior_matrix": embed_prior_matrix,
        "addl_embed_prior_matrix": addl_embed_prior_matrix,
        "repel_prior_matrix": repel_prior_matrix,
        "win_cluster_prior_matrix": win_cluster_prior_matrix,
        "addl_cluster_prior_matrix": addl_cluster_prior_matrix,
        "cluster_transition_matrix": cluster_transition_matrix,
        "embedding_prior_info": {
            "embed_dim": int(embed_system["embed_dim"]),
            "mean_confidence": float(embed_system["mean_confidence"]),
        },
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
                graph_prior_matrix=pipe["graph_prior_matrix"],
                embed_prior_matrix=pipe["embed_prior_matrix"],
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


def build_embedding_pattern_system(
    df: pd.DataFrame,
    decay: float = 0.992,
    context_window: int = 12,
    embed_dim: int = 12,
    temperature: float = 0.82,
    addl_strength: float = 0.30,
) -> Dict[str, np.ndarray | float | int]:
    """
    History-only vector embedding prior:
    - number vectors come from projected transition-context rows,
    - context comes from recent draws,
    - pattern priors come from single-slot and adjacent-pair transitions.
    """
    n = len(df)
    draw_win = df[WIN_COLS].values.astype(np.int32) - 1
    draw_addl = df["Addl No."].values.astype(np.int32) - 1

    embed_prior = np.zeros((n, 49), dtype=np.float32)
    addl_embed_prior = np.zeros((n, 49), dtype=np.float32)
    uniform = np.ones(49, dtype=np.float64) / 49.0

    freq = np.ones(49, dtype=np.float64)
    trans = np.ones((49, 49), dtype=np.float64)
    addl_trans = np.ones((49, 49), dtype=np.float64)
    single_slot_next = np.ones((6, 49, 49), dtype=np.float64) * 0.10
    pair_next: Dict[Tuple[int, int, int], np.ndarray] = {}

    d = int(np.clip(embed_dim, 6, 32))
    rng = np.random.default_rng(ACTIVE_SEED + 211)
    proj = rng.normal(0.0, 1.0, size=(49, d)).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12

    confidences: List[float] = []
    for t in range(n):
        if t == 0:
            embed_prior[t] = uniform.astype(np.float32)
            addl_embed_prior[t] = uniform.astype(np.float32)
        else:
            # Number "embeddings" from projected transition-context rows.
            context_rows = trans / (trans.sum(axis=1, keepdims=True) + 1e-12)
            num_vec = context_rows @ proj
            num_vec /= np.linalg.norm(num_vec, axis=1, keepdims=True) + 1e-12

            start = max(0, t - int(max(2, context_window)))
            recent_wins = draw_win[start:t].reshape(-1)
            recent_addl = draw_addl[start:t]
            if len(recent_wins) > 0:
                ctx = np.mean(num_vec[recent_wins], axis=0)
            else:
                ctx = np.mean(num_vec, axis=0)
            if len(recent_addl) > 0:
                ctx = ctx + 0.30 * np.mean(num_vec[recent_addl], axis=0)
            ctx_norm = float(np.linalg.norm(ctx)) + 1e-12
            sim = (num_vec @ ctx) / ctx_norm
            sim = sim - float(np.max(sim))
            emb_prob = np.exp(sim / float(np.clip(temperature, 0.55, 1.45)))
            emb_prob = normalize_prob(emb_prob)

            prev_unique = np.unique(draw_win[t - 1]).astype(np.int32)
            prev_addl = int(draw_addl[t - 1])
            prev_row = draw_win[t - 1]

            trans_prior = normalize_prob(np.mean(trans[prev_unique], axis=0))
            single_prior = normalize_prob(
                np.mean(single_slot_next[np.arange(6), prev_row], axis=0)
            )

            pair_acc = np.zeros(49, dtype=np.float64)
            pair_hits = 0
            for pos in range(5):
                key = (int(prev_row[pos]), int(prev_row[pos + 1]), pos)
                arr = pair_next.get(key)
                if arr is not None:
                    pair_acc += arr
                    pair_hits += 1
            if pair_hits > 0:
                pair_prior = normalize_prob(pair_acc)
            else:
                pair_prior = trans_prior

            freq_prior = normalize_prob(freq)
            win_prior = normalize_prob(
                0.34 * emb_prob
                + 0.28 * trans_prior
                + 0.20 * single_prior
                + 0.10 * pair_prior
                + 0.08 * freq_prior
            )
            addl_prior = normalize_prob(
                0.52 * normalize_prob(addl_trans[prev_addl])
                + 0.28 * emb_prob
                + 0.20 * freq_prior
            )

            embed_prior[t] = win_prior.astype(np.float32)
            addl_embed_prior[t] = addl_prior.astype(np.float32)
            confidences.append(float(np.max(win_prior)))

        # Decay old evidence, then consume current row evidence.
        freq *= decay
        trans *= decay
        addl_trans *= decay
        single_slot_next *= decay

        wins_t = np.unique(draw_win[t]).astype(np.int32)
        addl_t = int(draw_addl[t])
        freq[wins_t] += 1.0
        freq[addl_t] += float(addl_strength)

        if t > 0:
            prev_unique = np.unique(draw_win[t - 1]).astype(np.int32)
            prev_addl = int(draw_addl[t - 1])
            trans[np.ix_(prev_unique, wins_t)] += 1.0
            trans[prev_unique, addl_t] += 0.28
            addl_trans[prev_addl, wins_t] += 0.42
            addl_trans[prev_addl, addl_t] += 0.74

            prev_row = draw_win[t - 1]
            for pos in range(6):
                a = int(prev_row[pos])
                single_slot_next[pos, a, wins_t] += 1.0
                single_slot_next[pos, a, addl_t] += 0.22
            for pos in range(5):
                key = (int(prev_row[pos]), int(prev_row[pos + 1]), pos)
                arr = pair_next.get(key)
                if arr is None:
                    arr = np.ones(49, dtype=np.float64) * 1e-3
                    pair_next[key] = arr
                arr[wins_t] += 1.0
                arr[addl_t] += 0.22

    mean_conf = float(np.mean(confidences)) if confidences else float(1.0 / 49.0)
    return {
        "embed_prior": embed_prior.astype(np.float32),
        "addl_embed_prior": addl_embed_prior.astype(np.float32),
        "embed_dim": int(d),
        "mean_confidence": float(mean_conf),
    }


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


def normalized_entropy(prob: np.ndarray) -> float:
    p = normalize_prob(np.asarray(prob, dtype=np.float64))
    n = max(2, len(p))
    ent = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    return float(np.clip(ent / math.log(float(n)), 0.0, 1.0))


def temperature_scale_prob(prob: np.ndarray, temperature: float) -> np.ndarray:
    p = normalize_prob(np.asarray(prob, dtype=np.float64))
    t = float(np.clip(temperature, 0.55, 2.80))
    logits = np.log(np.clip(p, 1e-12, 1.0))
    logits = logits / t
    logits = logits - float(np.max(logits))
    ex = np.exp(logits)
    return normalize_prob(ex)


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


def build_combined_distributions(
    pred: Dict[str, np.ndarray],
    win_cluster_prior: np.ndarray,
    graph_prior: np.ndarray | None,
    embed_prior: np.ndarray | None,
    repel_prior: np.ndarray,
    weights: Dict[str, float],
    dynamic_blend: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    uniform = np.ones(49, dtype=np.float64) / 49.0
    graph_raw = normalize_prob(graph_prior if graph_prior is not None else uniform)
    embed_raw = normalize_prob(embed_prior if embed_prior is not None else graph_raw)
    cluster_raw = normalize_prob(win_cluster_prior if win_cluster_prior is not None else uniform)
    repel_raw = normalize_prob(repel_prior)

    slot_probs: List[np.ndarray] = []
    for i in range(1, 7):
        key = f"win_{i}"
        if key in pred and len(pred[key]) > 0:
            slot_probs.append(normalize_prob(pred[key][0]))
    if slot_probs:
        slot_mean = normalize_prob(np.mean(np.stack(slot_probs, axis=0), axis=0))
    else:
        slot_mean = uniform
    if "win_set" in pred and len(pred["win_set"]) > 0:
        win_set_prob = normalize_prob(pred["win_set"][0])
    else:
        win_set_prob = slot_mean
    model_soft_raw = normalize_prob(0.62 * win_set_prob + 0.38 * slot_mean)
    model_conf = float(max(np.max(model_soft_raw), np.max(graph_raw), np.max(embed_raw), np.max(cluster_raw)))

    if dynamic_blend:
        ent_model = normalized_entropy(model_soft_raw)
        ent_graph = normalized_entropy(graph_raw)
        ent_embed = normalized_entropy(embed_raw)
        ent_cluster = normalized_entropy(cluster_raw)

        model_soft_prob = temperature_scale_prob(model_soft_raw, 0.88 + 0.84 * ent_model)
        graph_prob = temperature_scale_prob(graph_raw, 0.92 + 0.76 * ent_graph)
        embed_prob = temperature_scale_prob(embed_raw, 0.90 + 0.80 * ent_embed)
        cluster_prob = temperature_scale_prob(cluster_raw, 0.94 + 0.74 * ent_cluster)

        model_rel = max(0.10, (1.0 - ent_model) ** 1.30)
        graph_rel = max(0.10, (1.0 - ent_graph) ** 1.34)
        embed_rel = max(0.10, (1.0 - ent_embed) ** 1.30)
        cluster_rel = max(0.10, (1.0 - ent_cluster) ** 1.25)

        model_weight = float(weights.get("model_soft", 0.0)) * (0.30 + 0.70 * model_rel)
        graph_weight = float(weights.get("graph", 0.0)) * (0.30 + 0.70 * graph_rel)
        embed_weight = float(weights.get("embed", 0.0)) * (0.30 + 0.70 * embed_rel)
        cluster_weight = float(weights.get("cluster", 0.0)) * (0.30 + 0.70 * cluster_rel)
        repel_scale = 0.26 + 1.06 * max(ent_model, ent_graph, ent_embed, ent_cluster)
    else:
        model_soft_prob = model_soft_raw
        graph_prob = graph_raw
        embed_prob = embed_raw
        cluster_prob = cluster_raw
        model_weight = float(weights.get("model_soft", 0.0))
        graph_weight = float(weights.get("graph", 0.0))
        embed_weight = float(weights.get("embed", 0.0))
        cluster_weight = float(weights.get("cluster", 0.0))
        repel_scale = 0.44 + 0.92 * (1.0 - model_conf)

    combined_win = (
        model_weight * model_soft_prob
        + graph_weight * graph_prob
        + embed_weight * embed_prob
        + cluster_weight * cluster_prob
        - (float(weights.get("repel", 0.0)) * repel_scale) * repel_raw
    )
    combined_win = normalize_prob(np.clip(combined_win, 1e-12, None))
    sharpen_t = float(np.clip(0.84 + 0.34 * (1.0 - model_conf), 0.70, 1.20))
    combined_win = np.power(np.clip(combined_win, 1e-12, None), 1.0 / sharpen_t)
    combined_win = normalize_prob(combined_win)

    addl_model = normalize_prob(pred["addl"][0])
    if dynamic_blend:
        ent_addl_model = normalized_entropy(addl_model)
        addl_model = temperature_scale_prob(addl_model, 0.92 + 0.78 * ent_addl_model)
    combined_addl = addl_model
    combined_addl = normalize_prob(np.clip(combined_addl, 1e-12, None))

    repel_prob = normalize_prob(repel_prior)
    return combined_win, combined_addl, model_soft_prob, graph_prob, embed_prob, cluster_prob, repel_prob, model_conf


def choose_addl_from_probs(combined_addl: np.ndarray, win_numbers: List[int]) -> int:
    addl_number = None
    wins = set(int(x) for x in win_numbers)
    for idx in np.argsort(combined_addl)[::-1]:
        candidate = int(idx + 1)
        if candidate not in wins:
            addl_number = candidate
            break
    if addl_number is None:
        addl_number = int(np.argmax(combined_addl) + 1)
    return int(addl_number)


def candidate_utility_from_probs(
    nums: List[int],
    combined_win: np.ndarray,
) -> float:
    idx = np.asarray(nums, dtype=np.int32) - 1
    p = np.clip(combined_win[idx], 1e-12, 1.0)
    # Tail-hit mode prefers concentrated high-probability sets.
    util = 2.9 * float(np.mean(np.log(p)))
    return float(util)


def gumbel_topk_set(prob: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    p = normalize_prob(prob)
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=len(p))
    g = -np.log(-np.log(u))
    scores = np.log(np.clip(p, 1e-12, 1.0)) + g
    idx = np.argpartition(scores, -k)[-k:]
    return sorted((idx + 1).astype(int).tolist())


def candidate_feature_vector(
    nums: List[int],
    combined_win: np.ndarray,
    model_soft_prob: np.ndarray,
    graph_prob: np.ndarray,
    embed_prob: np.ndarray,
    cluster_prob: np.ndarray,
    repel_prob: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(sorted(nums), dtype=np.int32)
    idx = arr - 1
    p = np.clip(combined_win[idx], 1e-12, 1.0)
    logp = np.log(p)
    gaps = np.diff(arr).astype(np.float64)
    gap_mean = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
    gap_std = float(np.std(gaps)) if len(gaps) > 0 else 0.0
    spread = float((arr[-1] - arr[0]) / 48.0) if len(arr) > 0 else 0.0
    dense_ratio = float(np.mean(gaps <= 2.0)) if len(gaps) > 0 else 0.0
    odd_balance = abs(float(np.sum(arr % 2 == 1)) - 3.0) / 3.0
    low_balance = abs(float(np.sum(arr <= 24)) - 3.0) / 3.0
    return np.asarray(
        [
            float(np.mean(logp)),
            float(np.min(logp)),
            float(np.max(logp)),
            float(np.std(logp)),
            float(np.mean(model_soft_prob[idx])),
            float(np.mean(graph_prob[idx])),
            float(np.mean(embed_prob[idx])),
            float(np.mean(cluster_prob[idx])),
            float(np.mean(repel_prob[idx])),
            spread,
            gap_mean / 8.0,
            gap_std / 6.0,
            dense_ratio,
            odd_balance,
            low_balance,
        ],
        dtype=np.float64,
    )


def candidate_expected_hit_signal(
    nums: List[int],
    combined_win: np.ndarray,
    model_soft_prob: np.ndarray,
    graph_prob: np.ndarray,
    embed_prob: np.ndarray,
    cluster_prob: np.ndarray,
) -> Tuple[float, float]:
    idx = np.asarray(sorted(nums), dtype=np.int32) - 1
    exp_hit = float(np.sum(np.clip(combined_win[idx], 1e-12, 1.0)))
    consensus = float(
        0.25 * np.mean(model_soft_prob[idx])
        + 0.25 * np.mean(graph_prob[idx])
        + 0.25 * np.mean(embed_prob[idx])
        + 0.25 * np.mean(cluster_prob[idx])
    )
    return exp_hit, consensus


def generate_candidate_sets(
    combined_win: np.ndarray,
    candidate_gumbel_samples: int = 28,
    candidate_max_pool: int = 48,
    rng: np.random.Generator | None = None,
) -> List[List[int]]:
    if rng is None:
        top_idx = np.argsort(combined_win)[-8:].astype(np.int64)
        top_p = np.round(combined_win[top_idx] * 1e6).astype(np.int64)
        seed_hint = int(np.dot(top_idx + 1, np.arange(1, len(top_idx) + 1, dtype=np.int64)) + np.sum(top_p))
        rng = np.random.default_rng(ACTIVE_SEED + 131 + (seed_hint % 1000003))
    cands: List[List[int]] = []
    cands.append(sorted([int(i + 1) for i in np.argsort(combined_win)[-6:].tolist()]))
    cands.append(select_win_numbers_with_repulsion(combined_win, count=6, p1=0.93, p2=0.97, p3=0.985))
    cands.append(select_win_numbers_with_repulsion(combined_win, count=6, p1=0.88, p2=0.94, p3=0.97))
    cands.append(select_win_numbers_with_repulsion(combined_win, count=6, p1=0.83, p2=0.90, p3=0.95))
    for temp in (0.78, 0.88, 1.05, 1.22):
        tprob = temperature_scale_prob(combined_win, temperature=temp)
        cands.append(sorted([int(i + 1) for i in np.argsort(tprob)[-6:].tolist()]))
    for _ in range(max(0, int(candidate_gumbel_samples))):
        cands.append(gumbel_topk_set(combined_win, k=6, rng=rng))

    out: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for c in cands:
        key = tuple(sorted(int(x) for x in c))
        if len(key) != 6:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
        if len(out) >= max(6, int(candidate_max_pool)):
            break
    return out


def repetition_penalty_score(
    key: Tuple[int, ...],
    recent_pred_sets: List[Tuple[int, ...]] | None,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
) -> float:
    if not recent_pred_sets:
        return 0.0
    window = int(max(1, anti_repeat_window))
    lam = float(max(0.0, anti_repeat_lambda))
    if lam <= 0.0:
        return 0.0

    key_set = set(int(x) for x in key)
    hist = recent_pred_sets[-window:]
    penalty = 0.0
    num_freq = np.zeros(49, dtype=np.float64)
    for rank, prev in enumerate(reversed(hist), start=1):
        decay = 1.0 / math.sqrt(float(rank))
        prev_set = set(int(x) for x in prev)
        overlap_n = len(key_set.intersection(prev_set))
        overlap = overlap_n / 6.0
        for n in prev_set:
            num_freq[int(n) - 1] += decay
        if key == prev:
            penalty += 2.20 * decay
        elif overlap_n >= 5:
            penalty += (1.45 + 0.35 * overlap_n) * decay
        elif overlap_n == 4:
            penalty += 1.65 * decay
        elif overlap_n == 3:
            penalty += 0.62 * decay
        elif overlap_n == 2:
            penalty += 0.18 * decay
        else:
            penalty += 0.06 * max(0.0, overlap - (1.0 / 6.0)) * decay

    idx = np.asarray([int(x) - 1 for x in key], dtype=np.int32)
    freq_norm = num_freq / (float(np.max(num_freq)) + 1e-12)
    num_pen = 0.42 * float(np.mean(freq_norm[idx]))
    penalty += num_pen
    return lam * penalty


def combine_prediction(
    pred: Dict[str, np.ndarray],
    win_cluster_prior: np.ndarray,
    graph_prior: np.ndarray | None,
    embed_prior: np.ndarray | None,
    repel_prior: np.ndarray,
    weights: Dict[str, float],
    dynamic_blend: bool = True,
    reward_reranker: RewardReranker | None = None,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
    expected_hit_rerank: bool = True,
    expected_hit_lambda: float = 2.20,
    expected_hit_synergy_lambda: float = 0.48,
    recent_pred_sets: List[Tuple[int, ...]] | None = None,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
    rng: np.random.Generator | None = None,
) -> Tuple[List[int], int]:
    combined_win, combined_addl, model_soft_prob, graph_prob, embed_prob, cluster_prob, repel_prob, model_conf = build_combined_distributions(
        pred=pred,
        win_cluster_prior=win_cluster_prior,
        graph_prior=graph_prior,
        embed_prior=embed_prior,
        repel_prior=repel_prior,
        weights=weights,
        dynamic_blend=dynamic_blend,
    )

    candidates = generate_candidate_sets(
        combined_win=combined_win,
        candidate_gumbel_samples=candidate_gumbel_samples,
        candidate_max_pool=candidate_max_pool,
        rng=rng,
    )
    scores: Dict[Tuple[int, ...], float] = {}
    for cand in candidates:
        key = tuple(sorted(cand))
        base_score = candidate_utility_from_probs(list(key), combined_win=combined_win)
        arr = np.asarray(key, dtype=np.int32)
        spread = float((arr[-1] - arr[0]) / 48.0)
        diversity_pen = float(candidate_diversity_lambda) * (1.0 - spread)
        score = base_score - diversity_pen
        score -= repetition_penalty_score(
            key=key,
            recent_pred_sets=recent_pred_sets,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
        )
        if bool(expected_hit_rerank):
            exp_hit, consensus = candidate_expected_hit_signal(
                nums=list(key),
                combined_win=combined_win,
                model_soft_prob=model_soft_prob,
                graph_prob=graph_prob,
                embed_prob=embed_prob,
                cluster_prob=cluster_prob,
            )
            score += float(expected_hit_lambda) * exp_hit
            score += float(expected_hit_synergy_lambda) * consensus
        if reward_reranker is not None:
            feat = candidate_feature_vector(
                nums=list(key),
                combined_win=combined_win,
                model_soft_prob=model_soft_prob,
                graph_prob=graph_prob,
                embed_prob=embed_prob,
                cluster_prob=cluster_prob,
                repel_prob=repel_prob,
            )
            score += float(reward_reranker.reward_scale) * reward_reranker.score(feat)
        scores[key] = score
    if candidates:
        plain_top = tuple(sorted(candidates[0]))
        if model_conf >= 0.28:
            scores[plain_top] = scores.get(plain_top, 0.0) + 0.06

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_key = ranked[0][0]
    if recent_pred_sets:
        hist = recent_pred_sets[-max(1, int(anti_repeat_window)) :]
        last_set = set(int(x) for x in hist[-1])
        strict_mode = float(anti_repeat_lambda) >= 0.60
        max_last_overlap = 2 if strict_mode else 3
        max_hist_overlap = 3 if strict_mode else 4
        for key, _sc in ranked:
            s = set(int(x) for x in key)
            overlap_last = len(s.intersection(last_set))
            max_overlap_hist = max(len(s.intersection(set(int(y) for y in prev))) for prev in hist)
            if overlap_last <= max_last_overlap and max_overlap_hist <= max_hist_overlap:
                best_key = key
                break
    win_numbers = list(best_key)
    addl_number = choose_addl_from_probs(combined_addl, win_numbers)
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


def emphasize_latest_weights(
    weights: np.ndarray,
    latest_n: int = 5,
    boost: float = 3.4,
    power: float = 1.9,
) -> np.ndarray:
    """
    Increase optimization pressure on the most recent samples.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) == 0:
        return w.astype(np.float32)
    latest_n = int(max(0, latest_n))
    boost = float(max(1.0, boost))
    if latest_n <= 0 or boost <= 1.0:
        return w.astype(np.float32)
    tail = int(min(latest_n, len(w)))
    start = int(len(w) - tail)
    ramp = np.linspace(0.35, 1.0, tail, dtype=np.float64)
    ramp = np.power(ramp, max(1.0, float(power)))
    out = w.copy()
    out[start:] *= 1.0 + (boost - 1.0) * ramp
    out = np.clip(out, 1e-8, None)
    out = out / (float(np.mean(out)) + 1e-12)
    return out.astype(np.float32)


def empty_hit_metrics() -> Dict[str, float]:
    return {
        "avg_win_hits": 0.0,
        "p_hit_ge2": 0.0,
        "p_hit_ge3": 0.0,
        "p_hit_ge4": 0.0,
        "addl_acc": 0.0,
        "hit_score": 0.0,
    }


def evaluate_hit_score(
    model: keras.Model,
    scaler_x: StandardScaler,
    X_seq: np.ndarray,
    eval_idx: np.ndarray,
    target_rows: np.ndarray,
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    graph_prior_matrix: np.ndarray,
    embed_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    weights: Dict[str, float],
    sample_weights: np.ndarray | None = None,
    cached_preds: Dict[str, np.ndarray] | None = None,
    dynamic_blend: bool = True,
    reward_reranker: RewardReranker | None = None,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
    expected_hit_rerank: bool = True,
    expected_hit_lambda: float = 2.20,
    expected_hit_synergy_lambda: float = 0.48,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
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
    recent_pred_sets: List[Tuple[int, ...]] = []
    for local_i, seq_i in enumerate(eval_idx):
        target_row = int(target_rows[seq_i])
        pred_pack = {k: v[local_i : local_i + 1] for k, v in preds.items()}

        pred_win, pred_addl = combine_prediction(
            pred=pred_pack,
            win_cluster_prior=win_cluster_prior_matrix[target_row],
            graph_prior=graph_prior_matrix[target_row],
            embed_prior=embed_prior_matrix[target_row],
            repel_prior=repel_prior_matrix[target_row],
            weights=weights,
            dynamic_blend=dynamic_blend,
            reward_reranker=reward_reranker,
            candidate_max_pool=candidate_max_pool,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_diversity_lambda=candidate_diversity_lambda,
            expected_hit_rerank=expected_hit_rerank,
            expected_hit_lambda=expected_hit_lambda,
            expected_hit_synergy_lambda=expected_hit_synergy_lambda,
            recent_pred_sets=recent_pred_sets,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
        )
        recent_pred_sets.append(tuple(sorted(int(x) for x in pred_win)))
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


def build_near_miss_variants(
    base_set: List[int],
    combined_win: np.ndarray,
    n_variants: int = 6,
) -> List[List[int]]:
    base = sorted(int(x) for x in base_set)
    pool = [int(i + 1) for i in np.argsort(combined_win)[::-1].tolist()]
    out: List[List[int]] = []
    if len(base) != 6:
        return out
    cursor = 0
    for replace_pos in range(6):
        trial = base.copy()
        while cursor < len(pool):
            cand = int(pool[cursor])
            cursor += 1
            if cand in trial:
                continue
            trial2 = trial.copy()
            trial2[replace_pos] = cand
            key = sorted(set(trial2))
            if len(key) == 6:
                out.append(key)
                break
        if len(out) >= max(1, int(n_variants)):
            break
    return out


def fit_reward_reranker(
    model: keras.Model,
    scaler_x: StandardScaler,
    X_seq: np.ndarray,
    eval_idx: np.ndarray,
    target_rows: np.ndarray,
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    graph_prior_matrix: np.ndarray,
    embed_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    blend_weights: Dict[str, float],
    dynamic_blend: bool = True,
    reward_a: float = 1.0,
    reward_b: float = 2.4,
    reward_c: float = 6.8,
    reward_window: int = 72,
    hardneg_boost: float = 0.9,
    ridge: float = 0.18,
    reward_scale: float = 0.55,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
) -> RewardReranker | None:
    if len(eval_idx) <= 0:
        return None
    eval_use = np.asarray(eval_idx, dtype=np.int32)
    if int(reward_window) > 0 and len(eval_use) > int(reward_window):
        eval_use = eval_use[-int(reward_window) :]
    if len(eval_use) <= 0:
        return None

    n_features = X_seq.shape[-1]
    X_eval = scaler_x.transform(X_seq[eval_use].reshape(-1, n_features)).reshape(
        len(eval_use), X_seq.shape[1], n_features
    ).astype(np.float32)
    preds = predict_outputs_dict(model, X_eval)
    rng = np.random.default_rng(ACTIVE_SEED + 991)

    feats: List[np.ndarray] = []
    rewards: List[float] = []
    sample_w: List[float] = []
    for local_i, seq_i in enumerate(eval_use):
        target_row = int(target_rows[int(seq_i)])
        pred_pack = {k: v[local_i : local_i + 1] for k, v in preds.items()}
        actual_set = set(df.loc[target_row, WIN_COLS].astype(int).tolist())
        combined_win, _combined_addl, model_soft_prob, graph_prob, embed_prob, cluster_prob, repel_prob, _mconf = build_combined_distributions(
            pred=pred_pack,
            win_cluster_prior=win_cluster_prior_matrix[target_row],
            graph_prior=graph_prior_matrix[target_row],
            embed_prior=embed_prior_matrix[target_row],
            repel_prior=repel_prior_matrix[target_row],
            weights=blend_weights,
            dynamic_blend=dynamic_blend,
        )
        cands = generate_candidate_sets(
            combined_win=combined_win,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_max_pool=candidate_max_pool,
            rng=rng,
        )
        if not cands:
            continue
        cands.extend(build_near_miss_variants(cands[0], combined_win=combined_win, n_variants=6))
        seen: set[Tuple[int, ...]] = set()
        for cand in cands:
            key = tuple(sorted(int(x) for x in cand))
            if key in seen or len(key) != 6:
                continue
            seen.add(key)
            hits = len(set(key).intersection(actual_set))
            reward = float(reward_a) * float(hits)
            if hits >= 3:
                reward += float(reward_b)
            if hits >= 4:
                reward += float(reward_c)
            arr = np.asarray(key, dtype=np.int32)
            spread = float((arr[-1] - arr[0]) / 48.0)
            reward -= float(candidate_diversity_lambda) * max(0.0, 0.55 - spread)
            feat = candidate_feature_vector(
                nums=list(key),
                combined_win=combined_win,
                model_soft_prob=model_soft_prob,
                graph_prob=graph_prob,
                embed_prob=embed_prob,
                cluster_prob=cluster_prob,
                repel_prob=repel_prob,
            )
            w = 1.0
            if hits in (1, 2):
                w += float(max(0.0, hardneg_boost))
            feats.append(feat)
            rewards.append(float(reward))
            sample_w.append(float(w))

    if len(feats) < 16:
        return None
    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(rewards, dtype=np.float64)
    w = np.asarray(sample_w, dtype=np.float64)
    w = np.clip(w, 1e-8, None)
    w = w / (float(np.mean(w)) + 1e-12)

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0) + 1e-6
    Xn = (X - mu) / sd
    XtW = Xn.T * w
    reg = float(max(1e-6, ridge))
    A = XtW @ Xn + reg * np.eye(Xn.shape[1], dtype=np.float64)
    b = XtW @ y
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(A) @ b
    intercept = float(np.dot(w, (y - Xn @ coef)) / (float(np.sum(w)) + 1e-12))
    return RewardReranker(
        feature_mean=mu.astype(np.float64),
        feature_std=sd.astype(np.float64),
        coef=coef.astype(np.float64),
        intercept=float(intercept),
        reward_scale=float(np.clip(reward_scale, 0.05, 3.0)),
    )


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
        {"model_soft": 0.56, "graph": 0.92, "embed": 0.44, "cluster": 0.66, "repel": 0.00, "addl_model": 1.00},
        {"model_soft": 0.76, "graph": 0.82, "embed": 0.58, "cluster": 0.60, "repel": 0.03, "addl_model": 1.00},
        {"model_soft": 0.92, "graph": 0.62, "embed": 0.78, "cluster": 0.48, "repel": 0.07, "addl_model": 1.00},
        {"model_soft": 0.68, "graph": 1.00, "embed": 0.25, "cluster": 0.72, "repel": 0.12, "addl_model": 1.00},
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
    for k in WIN_BLEND_KEYS + ADDL_BLEND_KEYS:
        out[k] = float(out.get(k, min_value))
    out["repel"] = float(out.get("repel", 0.0))
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
    graph_prior_matrix: np.ndarray,
    embed_prior_matrix: np.ndarray,
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
    dynamic_blend: bool = True,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
    expected_hit_rerank: bool = True,
    expected_hit_lambda: float = 2.20,
    expected_hit_synergy_lambda: float = 0.48,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
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
                    graph_prior_matrix=graph_prior_matrix,
                    embed_prior_matrix=embed_prior_matrix,
                    addl_cluster_prior_matrix=addl_cluster_prior_matrix,
                    repel_prior_matrix=repel_prior_matrix,
                    weights=blend_weights,
                    sample_weights=eval_weights,
                    dynamic_blend=dynamic_blend,
                    candidate_max_pool=candidate_max_pool,
                    candidate_gumbel_samples=candidate_gumbel_samples,
                    candidate_diversity_lambda=candidate_diversity_lambda,
                    expected_hit_rerank=expected_hit_rerank,
                    expected_hit_lambda=expected_hit_lambda,
                    expected_hit_synergy_lambda=expected_hit_synergy_lambda,
                    anti_repeat_window=anti_repeat_window,
                    anti_repeat_lambda=anti_repeat_lambda,
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
                    graph_prior_matrix=graph_prior_matrix,
                    embed_prior_matrix=embed_prior_matrix,
                    addl_cluster_prior_matrix=addl_cluster_prior_matrix,
                    repel_prior_matrix=repel_prior_matrix,
                    weights=blend_weights,
                    dynamic_blend=dynamic_blend,
                    candidate_max_pool=candidate_max_pool,
                    candidate_gumbel_samples=candidate_gumbel_samples,
                    candidate_diversity_lambda=candidate_diversity_lambda,
                    expected_hit_rerank=expected_hit_rerank,
                    expected_hit_lambda=expected_hit_lambda,
                    expected_hit_synergy_lambda=expected_hit_synergy_lambda,
                    anti_repeat_window=anti_repeat_window,
                    anti_repeat_lambda=anti_repeat_lambda,
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


def model_config_key(cfg: ModelConfig) -> Tuple[int, int, int, int, float, float]:
    return (
        int(cfg.seq_len),
        int(cfg.lstm_units),
        int(cfg.gru_units),
        int(cfg.dense_units),
        float(cfg.dropout),
        float(cfg.lr),
    )


def run_tuning_multifidelity(
    trial_configs: List[ModelConfig],
    seq_cache: Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]],
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    graph_prior_matrix: np.ndarray,
    embed_prior_matrix: np.ndarray,
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
    dynamic_blend: bool = True,
    enable_multifidelity: bool = True,
    stage1_epochs: int = 5,
    keep_ratio: float = 0.45,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
    expected_hit_rerank: bool = True,
    expected_hit_lambda: float = 2.20,
    expected_hit_synergy_lambda: float = 0.48,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
) -> pd.DataFrame:
    if (not enable_multifidelity) or len(trial_configs) <= 3:
        return run_tuning(
            trial_configs=trial_configs,
            seq_cache=seq_cache,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            blend_weights=blend_weights,
            tune_folds=tune_folds,
            tune_epochs=tune_epochs,
            focus_last_n=focus_last_n,
            batch_size=batch_size,
            hit_score_focused=hit_score_focused,
            steps_per_execution=steps_per_execution,
            cache_dataset=cache_dataset,
            train_recency_weighted=train_recency_weighted,
            dynamic_blend=dynamic_blend,
            candidate_max_pool=candidate_max_pool,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_diversity_lambda=candidate_diversity_lambda,
            expected_hit_rerank=expected_hit_rerank,
            expected_hit_lambda=expected_hit_lambda,
            expected_hit_synergy_lambda=expected_hit_synergy_lambda,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
        )

    s1_epochs = int(max(1, min(int(stage1_epochs), int(max(1, tune_epochs)))))
    s1_folds = int(np.clip(tune_folds, 1, 2))
    print(
        f"[TUNE-MF] stage1: trials={len(trial_configs)} folds={s1_folds} epochs={s1_epochs}; "
        f"stage2 epochs={tune_epochs}"
    )
    stage1 = run_tuning(
        trial_configs=trial_configs,
        seq_cache=seq_cache,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        graph_prior_matrix=graph_prior_matrix,
        embed_prior_matrix=embed_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=blend_weights,
        tune_folds=s1_folds,
        tune_epochs=s1_epochs,
        focus_last_n=focus_last_n,
        batch_size=batch_size,
        hit_score_focused=hit_score_focused,
        steps_per_execution=steps_per_execution,
        cache_dataset=cache_dataset,
        train_recency_weighted=train_recency_weighted,
        dynamic_blend=dynamic_blend,
        candidate_max_pool=candidate_max_pool,
        candidate_gumbel_samples=candidate_gumbel_samples,
        candidate_diversity_lambda=candidate_diversity_lambda,
        expected_hit_rerank=expected_hit_rerank,
        expected_hit_lambda=expected_hit_lambda,
        expected_hit_synergy_lambda=expected_hit_synergy_lambda,
        anti_repeat_window=anti_repeat_window,
        anti_repeat_lambda=anti_repeat_lambda,
    )
    if stage1.empty:
        return stage1

    keep_n = int(max(2, min(len(stage1), math.ceil(len(stage1) * float(np.clip(keep_ratio, 0.15, 0.9))))))
    top = stage1.head(keep_n)
    keep_keys = set(
        (
            int(r.seq_len),
            int(r.lstm_units),
            int(r.gru_units),
            int(r.dense_units),
            float(r.dropout),
            float(r.lr),
        )
        for r in top.itertuples(index=False)
    )
    shortlisted = [cfg for cfg in trial_configs if model_config_key(cfg) in keep_keys]
    if len(shortlisted) <= 0:
        shortlisted = trial_configs[: keep_n]
    print(f"[TUNE-MF] stage2 shortlisted {len(shortlisted)}/{len(trial_configs)} configs")
    return run_tuning(
        trial_configs=shortlisted,
        seq_cache=seq_cache,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        graph_prior_matrix=graph_prior_matrix,
        embed_prior_matrix=embed_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=blend_weights,
        tune_folds=tune_folds,
        tune_epochs=tune_epochs,
        focus_last_n=focus_last_n,
        batch_size=batch_size,
        hit_score_focused=hit_score_focused,
        steps_per_execution=steps_per_execution,
        cache_dataset=cache_dataset,
        train_recency_weighted=train_recency_weighted,
        dynamic_blend=dynamic_blend,
        candidate_max_pool=candidate_max_pool,
        candidate_gumbel_samples=candidate_gumbel_samples,
        candidate_diversity_lambda=candidate_diversity_lambda,
        expected_hit_rerank=expected_hit_rerank,
        expected_hit_lambda=expected_hit_lambda,
        expected_hit_synergy_lambda=expected_hit_synergy_lambda,
        anti_repeat_window=anti_repeat_window,
        anti_repeat_lambda=anti_repeat_lambda,
    )


def calibrate_blend_weights_by_component_performance(
    model: keras.Model,
    scaler_x: StandardScaler,
    X_seq: np.ndarray,
    eval_idx: np.ndarray,
    target_rows: np.ndarray,
    df: pd.DataFrame,
    win_cluster_prior_matrix: np.ndarray,
    graph_prior_matrix: np.ndarray,
    embed_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    base_weights: Dict[str, float],
    sample_weights: np.ndarray | None = None,
    cached_preds: Dict[str, np.ndarray] | None = None,
    dynamic_blend: bool = True,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
    expected_hit_rerank: bool = True,
    expected_hit_lambda: float = 2.20,
    expected_hit_synergy_lambda: float = 0.48,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
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
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=w,
            sample_weights=sample_weights,
            cached_preds=cached_preds,
            dynamic_blend=dynamic_blend,
            candidate_max_pool=candidate_max_pool,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_diversity_lambda=candidate_diversity_lambda,
            expected_hit_rerank=expected_hit_rerank,
            expected_hit_lambda=expected_hit_lambda,
            expected_hit_synergy_lambda=expected_hit_synergy_lambda,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
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
        for wk in win_keys:
            w[wk] = float(base_weights.get(wk, 0.0))
        w[key] = 1.0
        metrics = evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=w,
            sample_weights=sample_weights,
            cached_preds=cached_preds,
            dynamic_blend=dynamic_blend,
            candidate_max_pool=candidate_max_pool,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_diversity_lambda=candidate_diversity_lambda,
            expected_hit_rerank=expected_hit_rerank,
            expected_hit_lambda=expected_hit_lambda,
            expected_hit_synergy_lambda=expected_hit_synergy_lambda,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
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
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=w,
            sample_weights=sample_weights,
            cached_preds=cached_preds,
            dynamic_blend=dynamic_blend,
            candidate_max_pool=candidate_max_pool,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_diversity_lambda=candidate_diversity_lambda,
            expected_hit_rerank=expected_hit_rerank,
            expected_hit_lambda=expected_hit_lambda,
            expected_hit_synergy_lambda=expected_hit_synergy_lambda,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
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
    graph_prior_matrix: np.ndarray,
    embed_prior_matrix: np.ndarray,
    addl_cluster_prior_matrix: np.ndarray,
    repel_prior_matrix: np.ndarray,
    blend_weights: Dict[str, float],
    backtest_folds: int,
    backtest_epochs: int,
    focus_last_n: int,
    latest_priority_n: int,
    latest_priority_boost: float,
    latest_priority_lambda: float,
    latest_priority_addl_lambda: float,
    batch_size: int,
    local_random_candidates: int,
    optimize_avg: bool,
    backtest_restarts: int = 1,
    restart_ensemble_topk: int = 1,
    steps_per_execution: int = 0,
    cache_dataset: bool = False,
    train_recency_weighted: bool = False,
    dynamic_blend: bool = True,
    reward_reranker: RewardReranker | None = None,
    candidate_max_pool: int = 48,
    candidate_gumbel_samples: int = 28,
    candidate_diversity_lambda: float = 0.11,
    expected_hit_rerank: bool = True,
    expected_hit_lambda: float = 2.20,
    expected_hit_synergy_lambda: float = 0.48,
    anti_repeat_window: int = 10,
    anti_repeat_lambda: float = 1.25,
    no_retrain: bool = False,
    inference_model: keras.Model | None = None,
    inference_scaler: StandardScaler | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n_samples = len(X_seq)
    records: List[Dict[str, object]] = []

    def _row_draw(target_row: int) -> object:
        v = df.loc[target_row, "Draw"]
        return int(v) if not pd.isna(v) else ""

    def _row_date(target_row: int) -> str:
        v = df.loc[target_row, "Date"]
        return v.strftime("%Y-%m-%d") if not pd.isna(v) else ""

    def _actual_values(target_row: int) -> Tuple[List[int], int]:
        actual_win = df.loc[target_row, WIN_COLS].astype(int).tolist()
        actual_addl = int(df.loc[target_row, "Addl No."])
        return actual_win, actual_addl

    def _cluster_state(target_row: int) -> Tuple[int, int]:
        hist_labels = cluster_labels[:target_row]
        if len(hist_labels) > 1:
            _tm, last_cluster, next_cluster = transition_from_labels(hist_labels, best_k)
            return int(last_cluster), int(next_cluster)
        fallback = int(cluster_labels[max(0, target_row - 1)])
        return fallback, fallback

    def _build_payload(fold: int, target_row: int, pred_pack: Dict[str, np.ndarray]) -> Dict[str, object]:
        last_cluster, next_cluster = _cluster_state(target_row)
        actual_win, actual_addl = _actual_values(target_row)
        return {
            "fold": fold,
            "target_row": target_row,
            "draw": _row_draw(target_row),
            "date": _row_date(target_row),
            "cluster_last": last_cluster,
            "cluster_pred_next": next_cluster,
            "pred_pack": pred_pack,
            "win_cluster_prior": win_cluster_prior_matrix[target_row],
            "graph_prior": graph_prior_matrix[target_row],
            "embed_prior": embed_prior_matrix[target_row],
            "repel_prior": repel_prior_matrix[target_row],
            "actual_win": actual_win,
            "actual_addl": actual_addl,
        }

    def _predict_for_target_row(
        pred_pack: Dict[str, np.ndarray],
        target_row: int,
        weights: Dict[str, float],
        recent_pred_sets: List[Tuple[int, ...]] | None = None,
    ) -> Tuple[List[int], int]:
        return combine_prediction(
            pred=pred_pack,
            win_cluster_prior=win_cluster_prior_matrix[target_row],
            graph_prior=graph_prior_matrix[target_row],
            embed_prior=embed_prior_matrix[target_row],
            repel_prior=repel_prior_matrix[target_row],
            weights=weights,
            dynamic_blend=dynamic_blend,
            reward_reranker=reward_reranker,
            candidate_max_pool=candidate_max_pool,
            candidate_gumbel_samples=candidate_gumbel_samples,
            candidate_diversity_lambda=candidate_diversity_lambda,
            expected_hit_rerank=expected_hit_rerank,
            expected_hit_lambda=expected_hit_lambda,
            expected_hit_synergy_lambda=expected_hit_synergy_lambda,
            recent_pred_sets=recent_pred_sets,
            anti_repeat_window=anti_repeat_window,
            anti_repeat_lambda=anti_repeat_lambda,
        )

    def _record_from_prediction(
        fold: int,
        target_row: int,
        pred_win: List[int],
        pred_addl: int,
        actual_win: List[int],
        actual_addl: int,
        cluster_last: int,
        cluster_pred_next: int,
    ) -> Dict[str, object]:
        win_hits, addl_hit = summarize_hits(pred_win, pred_addl, actual_win, int(actual_addl))
        return {
            "fold": fold,
            "target_row": target_row,
            "draw": _row_draw(target_row),
            "date": _row_date(target_row),
            "pred_win": " ".join(str(x) for x in pred_win),
            "actual_win": " ".join(str(x) for x in sorted(actual_win)),
            "pred_addl": pred_addl,
            "actual_addl": int(actual_addl),
            "win_hits": win_hits,
            "addl_hit": addl_hit,
            "cluster_last": cluster_last,
            "cluster_pred_next": cluster_pred_next,
        }

    if focus_last_n > 0:
        # Strict one-step walk-forward on most recent N draws only.
        test_seq = recent_seq_indices(n_samples, focus_last_n, min_train=max(120, config.seq_len * 4))
        payloads: List[Dict[str, object]] = []
        if bool(no_retrain) and inference_model is not None and inference_scaler is not None and len(test_seq) > 0:
            n_features = X_seq.shape[-1]
            X_test = inference_scaler.transform(X_seq[test_seq].reshape(-1, n_features)).reshape(
                len(test_seq), X_seq.shape[1], n_features
            ).astype(np.float32)
            preds = predict_outputs_dict(inference_model, X_test)
            for pos, seq_i in enumerate(test_seq, start=1):
                target_row = int(target_rows[seq_i])
                pred_pack = {k: v[pos - 1 : pos] for k, v in preds.items()}
                payloads.append(_build_payload(pos, target_row, pred_pack))
        else:
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
                        graph_prior_matrix=graph_prior_matrix,
                        embed_prior_matrix=embed_prior_matrix,
                        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
                        repel_prior_matrix=repel_prior_matrix,
                        weights=blend_weights,
                        sample_weights=val_sw,
                        dynamic_blend=dynamic_blend,
                        reward_reranker=reward_reranker,
                        candidate_max_pool=candidate_max_pool,
                        candidate_gumbel_samples=candidate_gumbel_samples,
                        candidate_diversity_lambda=candidate_diversity_lambda,
                        expected_hit_rerank=expected_hit_rerank,
                        expected_hit_lambda=expected_hit_lambda,
                        expected_hit_synergy_lambda=expected_hit_synergy_lambda,
                        anti_repeat_window=anti_repeat_window,
                        anti_repeat_lambda=anti_repeat_lambda,
                    )
                    restart_score = float(
                        3.2 * val_metrics["avg_win_hits"]
                        + 2.8 * val_metrics["p_hit_ge3"]
                        + 7.2 * val_metrics["p_hit_ge4"]
                        + 0.3 * val_metrics["p_hit_ge2"]
                    )
                    restart_pool.append((restart_score, pred_one))

                if not restart_pool:
                    continue
                restart_pool.sort(key=lambda x: x[0], reverse=True)
                chosen = restart_pool[:top_k]
                pred_pack = {
                    k: np.mean(np.stack([p[k] for _, p in chosen], axis=0), axis=0)
                    for k in chosen[0][1].keys()
                }
                target_row = int(target_rows[seq_i])
                payloads.append(_build_payload(pos, target_row, pred_pack))

        if payloads:
            best_weights = dict(blend_weights)
            best_score = -1e12
            payload_draw = np.asarray([item["draw"] for item in payloads], dtype=object)
            payload_date = np.asarray([item["date"] for item in payloads], dtype=object)
            payload_order = np.arange(len(payloads), dtype=np.float64)
            payload_recency_w = recency_weights_from_draw_date(payload_draw, payload_date, payload_order)
            payload_recency_w = emphasize_latest_weights(
                payload_recency_w,
                latest_n=latest_priority_n,
                boost=latest_priority_boost,
                power=1.9,
            )
            local_random = int(max(64, local_random_candidates if local_random_candidates > 0 else (560 if focus_last_n > 0 else 260)))
            for cand in generate_backtest_blend_candidates(blend_weights, num_random=local_random):
                hits: List[int] = []
                addl_hits: List[int] = []
                recent_pred_sets: List[Tuple[int, ...]] = []
                for item in payloads:
                    pred_win, pred_addl = _predict_for_target_row(
                        pred_pack=item["pred_pack"],
                        target_row=int(item["target_row"]),
                        weights=cand,
                        recent_pred_sets=recent_pred_sets,
                    )
                    recent_pred_sets.append(tuple(sorted(int(x) for x in pred_win)))
                    h, a = summarize_hits(pred_win, pred_addl, item["actual_win"], int(item["actual_addl"]))
                    hits.append(h)
                    addl_hits.append(a)

                hits_arr = np.asarray(hits, dtype=np.float32)
                addl_arr = np.asarray(addl_hits, dtype=np.float32)
                m = hit_metrics_from_arrays(hits_arr, addl_arr, sample_weights=payload_recency_w)
                tail_n = int(min(max(0, latest_priority_n), len(hits_arr)))
                if tail_n > 0:
                    tail_m = hit_metrics_from_arrays(hits_arr[-tail_n:], addl_arr[-tail_n:])
                else:
                    tail_m = empty_hit_metrics()
                if bool(optimize_avg):
                    score = (
                        210.0 * float(m["avg_win_hits"])
                        + 150.0 * float(m["p_hit_ge2"])
                        + 95.0 * float(m["p_hit_ge3"])
                        + 52.0 * float(m["p_hit_ge4"])
                        + 6.0 * float(m["addl_acc"])
                    )
                    # Strongly favor the latest N draws (especially latest 5 by default).
                    tail_score = (
                        300.0 * float(tail_m["avg_win_hits"])
                        + 140.0 * float(tail_m["p_hit_ge2"])
                        + 120.0 * float(tail_m["p_hit_ge3"])
                        + 84.0 * float(tail_m["p_hit_ge4"])
                    )
                    score += float(latest_priority_lambda) * tail_score
                    score += float(latest_priority_addl_lambda) * (120.0 * float(tail_m["addl_acc"]))
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

            recent_pred_sets: List[Tuple[int, ...]] = []
            for item in payloads:
                pred_win, pred_addl = _predict_for_target_row(
                    pred_pack=item["pred_pack"],
                    target_row=int(item["target_row"]),
                    weights=best_weights,
                    recent_pred_sets=recent_pred_sets,
                )
                recent_pred_sets.append(tuple(sorted(int(x) for x in pred_win)))
                records.append(
                    _record_from_prediction(
                        fold=int(item["fold"]),
                        target_row=int(item["target_row"]),
                        pred_win=pred_win,
                        pred_addl=pred_addl,
                        actual_win=item["actual_win"],
                        actual_addl=int(item["actual_addl"]),
                        cluster_last=int(item["cluster_last"]),
                        cluster_pred_next=int(item["cluster_pred_next"]),
                    )
                )
    else:
        fold_count = max(1, int(backtest_folds))
        min_total = fold_count * 24
        backtest_total = max(min_total, int(n_samples * 0.2))
        backtest_start = max(1, n_samples - backtest_total)
        block = max(1, (n_samples - backtest_start) // fold_count)

        for fold in range(fold_count):
            test_start = backtest_start + fold * block
            test_end = backtest_start + (fold + 1) * block if fold < fold_count - 1 else n_samples
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

            recent_pred_sets: List[Tuple[int, ...]] = []
            for local_i, seq_i in enumerate(test_idx):
                pred_pack = {k: v[local_i : local_i + 1] for k, v in preds.items()}
                target_row = int(target_rows[seq_i])
                last_cluster, next_cluster = _cluster_state(target_row)
                pred_win, pred_addl = _predict_for_target_row(
                    pred_pack=pred_pack,
                    target_row=target_row,
                    weights=blend_weights,
                    recent_pred_sets=recent_pred_sets,
                )
                recent_pred_sets.append(tuple(sorted(int(x) for x in pred_win)))
                actual_win, actual_addl = _actual_values(target_row)

                records.append(
                    _record_from_prediction(
                        fold=fold + 1,
                        target_row=target_row,
                        pred_win=pred_win,
                        pred_addl=pred_addl,
                        actual_win=actual_win,
                        actual_addl=actual_addl,
                        cluster_last=last_cluster,
                        cluster_pred_next=next_cluster,
                    )
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
    recency_w = emphasize_latest_weights(
        recency_w,
        latest_n=latest_priority_n,
        boost=latest_priority_boost,
        power=1.9,
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
    transition_matrix: np.ndarray,
    dashboard: Dict[str, object],
    focus_last_n: int,
    feature_group_weights: Dict[str, float],
    feature_weight_tuning_df: pd.DataFrame,
    blend_component_df: pd.DataFrame,
    calibrated_blend: Dict[str, float],
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
    if not metrics_df.empty:
        for c in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[c]):
                metrics_df[c] = metrics_df[c].map(lambda x: f"{x:.6f}" if isinstance(x, (float, np.floating)) else x)
    blend_pills: List[str] = []
    for k in WIN_BLEND_KEYS + ["repel"] + ADDL_BLEND_KEYS:
        if k in calibrated_blend:
            blend_pills.append(f'<div class="pill"><strong>{k}</strong><br>{float(calibrated_blend[k]):.6f}</div>')
    blend_pills_html = "".join(blend_pills)
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
      <div class="pill"><strong>Predicted Win</strong><br>{prediction["hybrid_win_numbers"]}</div>
      <div class="pill"><strong>Predicted Addl</strong><br>{prediction["hybrid_addl_number"]}</div>
      <div class="pill"><strong>Cluster Last -> Next</strong><br>{prediction["cluster_last"]} -> {prediction["cluster_next"]}</div>
      <div class="pill"><strong>Predicted Sum / Mean</strong><br>{prediction["pred_sum"]} / {prediction["pred_mean"]:.2f}</div>
      <div class="pill"><strong>Low / High</strong><br>{prediction["low_count"]} / {prediction["high_count"]}</div>
      <div class="pill"><strong>Odd / Even</strong><br>{prediction["odd_count"]} / {prediction["even_count"]}</div>
    </div>
    <p class="small">Final prediction uses Hybrid DL blend.</p>
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
      <div class="pill"><strong>Graph Co-Occurrence</strong><br>{feature_group_weights["graph"]:.4f}</div>
      <div class="pill"><strong>Vector Embedding</strong><br>{feature_group_weights["embed"]:.4f}</div>
    </div>
    {"<h3>Feature Weight Tuning Trials</h3><div class='grid-wrap'>" + html_table(feature_weight_show) + "</div>" if not feature_weight_show.empty else "<p class='small'>Feature weight tuning was disabled for this run.</p>"}
  </div>

  <div class="card">
    <h2>Adaptive Blend Weights</h2>
    <div class="kpi">
      {blend_pills_html}
    </div>
    {"<h3>Component Performance (weight basis)</h3><div class='grid-wrap'>" + html_table(blend_component_show) + "</div>" if not blend_component_show.empty else "<p class='small'>Adaptive component calibration was not available.</p>"}
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

    Plotly.newPlot('transition_heat', [{{
      z: D.cluster.transition_z,
      x: D.cluster.transition_x,
      y: D.cluster.transition_y,
      type: 'heatmap',
      colorscale: 'Magma'
    }}], {{...baseLayout, title:'Cluster Transition Probability Matrix'}}, {{responsive:true}});

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

  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    stage_times: Dict[str, float] = {}
    t_stage = time.perf_counter()

    def _mark_stage(name: str) -> None:
        nonlocal t_stage
        now = time.perf_counter()
        stage_times[name] = float(now - t_stage)
        t_stage = now

    def _profile_or_clip(
        value: float,
        *,
        default: float,
        profile: float,
        lo: float | None = None,
        hi: float | None = None,
    ) -> float:
        v = float(value)
        if abs(v - float(default)) < 1e-9:
            return float(profile)
        if lo is not None and hi is not None:
            return float(np.clip(v, lo, hi))
        if lo is not None:
            return float(max(v, lo))
        if hi is not None:
            return float(min(v, hi))
        return v

    def _profile_or_floor_int(value: int, *, default_max: int, profile: int, floor: int) -> int:
        v = int(value)
        if v <= int(default_max):
            return int(profile)
        return int(max(v, floor))

    def _enforce_latest_priority_min(min_boost: float, min_lambda: float, min_addl_lambda: float) -> None:
        args.latest_priority_n = max(int(args.latest_priority_n), 5)
        args.latest_priority_boost = max(float(args.latest_priority_boost), float(min_boost))
        args.latest_priority_lambda = max(float(args.latest_priority_lambda), float(min_lambda))
        args.latest_priority_addl_lambda = max(float(args.latest_priority_addl_lambda), float(min_addl_lambda))

    if bool(args.hit_score_focused):
        prev_focus = int(args.focus_last_n)
        args.focus_last_n = 24 if int(args.focus_last_n) <= 0 else max(int(args.focus_last_n), 24)
        if bool(args.sweep_mode):
            args.tune_trials = max(int(args.tune_trials), 8)
            args.tune_epochs = max(int(args.tune_epochs), 8)
            args.tune_mf_stage1_epochs = max(3, min(int(args.tune_mf_stage1_epochs), int(args.tune_epochs)))
            args.tune_mf_keep_ratio = float(np.clip(args.tune_mf_keep_ratio, 0.20, 0.70))
            args.multi_restarts = max(int(args.multi_restarts), 2)
            args.final_epochs = max(int(args.final_epochs), 42)
            args.backtest_folds = max(int(args.backtest_folds), 10)
            args.backtest_epochs = max(int(args.backtest_epochs), 12)
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
            _enforce_latest_priority_min(2.8, 1.6, 1.2)
        else:
            args.tune_trials = max(int(args.tune_trials), 14)
            args.tune_epochs = max(int(args.tune_epochs), 10)
            args.tune_mf_stage1_epochs = max(4, min(int(args.tune_mf_stage1_epochs), int(args.tune_epochs)))
            args.tune_mf_keep_ratio = float(np.clip(args.tune_mf_keep_ratio, 0.25, 0.70))
            args.multi_restarts = max(int(args.multi_restarts), 5)
            args.final_epochs = max(int(args.final_epochs), 64)
            args.backtest_folds = max(int(args.backtest_folds), 10)
            args.backtest_epochs = max(int(args.backtest_epochs), 20)
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
            _enforce_latest_priority_min(3.4, 2.2, 1.6)
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
            f"dyn_blend={'Y' if args.uncertainty_dynamic_blend else 'N'}, "
            f"cand_pool={args.candidate_max_pool}, gumbel={args.candidate_gumbel_samples}, "
            f"reward={'Y' if args.reward_rerank else 'N'}, "
            f"anti_repeat_w={args.anti_repeat_window}, anti_repeat_l={float(args.anti_repeat_lambda):.2f}, "
            f"latestN={args.latest_priority_n}, latestBoost={float(args.latest_priority_boost):.2f}, "
            f"latestL={float(args.latest_priority_lambda):.2f}, latestAddlL={float(args.latest_priority_addl_lambda):.2f}"
        )
    if bool(args.avg_hit_focused):
        prev_backtest_opt = bool(args.backtest_optimize_avg)
        args.backtest_optimize_avg = True
        # Anchor avg-focus defaults to best observed profile:
        # run_01_L01_base_lowanti (avg_win_hits=1.5000).
        args.w_graph = _profile_or_clip(args.w_graph, default=1.6, profile=1.90, lo=1.60)
        args.w_cluster = _profile_or_clip(args.w_cluster, default=1.35, profile=1.00, lo=0.70, hi=1.35)
        args.reward_a = _profile_or_clip(args.reward_a, default=1.0, profile=2.00, lo=1.70, hi=2.40)
        args.reward_b = _profile_or_clip(args.reward_b, default=2.4, profile=2.30, lo=1.80, hi=2.80)
        args.reward_c = _profile_or_clip(args.reward_c, default=6.8, profile=4.60, lo=3.60, hi=5.80)
        args.reward_scale = _profile_or_clip(args.reward_scale, default=0.55, profile=0.55, lo=0.45, hi=0.70)
        args.anti_repeat_window = _profile_or_floor_int(args.anti_repeat_window, default_max=10, profile=12, floor=10)
        args.anti_repeat_lambda = _profile_or_clip(args.anti_repeat_lambda, default=1.25, profile=0.55, lo=0.45, hi=1.10)
        args.candidate_diversity_lambda = _profile_or_clip(
            args.candidate_diversity_lambda, default=0.11, profile=0.08, lo=0.05, hi=0.16
        )
        args.candidate_max_pool = _profile_or_floor_int(args.candidate_max_pool, default_max=48, profile=72, floor=48)
        args.candidate_gumbel_samples = _profile_or_floor_int(args.candidate_gumbel_samples, default_max=28, profile=52, floor=28)
        args.expected_hit_rerank = True
        args.expected_hit_lambda = _profile_or_clip(args.expected_hit_lambda, default=2.20, profile=2.80, lo=2.20, hi=3.40)
        args.expected_hit_synergy_lambda = _profile_or_clip(
            args.expected_hit_synergy_lambda, default=0.48, profile=0.72, lo=0.40, hi=1.00
        )
        args.latest_priority_n = _profile_or_floor_int(args.latest_priority_n, default_max=5, profile=5, floor=3)
        args.latest_priority_boost = _profile_or_clip(
            args.latest_priority_boost, default=3.4, profile=4.2, lo=2.6, hi=7.0
        )
        args.latest_priority_lambda = _profile_or_clip(
            args.latest_priority_lambda, default=2.2, profile=2.9, lo=1.4, hi=5.0
        )
        args.latest_priority_addl_lambda = _profile_or_clip(
            args.latest_priority_addl_lambda, default=1.6, profile=2.3, lo=0.9, hi=4.0
        )
        print(
            "[AVG-FOCUS] enabled: "
            f"bt_opt_avg {prev_backtest_opt}->{args.backtest_optimize_avg}, "
            f"w_graph={args.w_graph:.2f}, w_cluster={args.w_cluster:.2f}, "
            f"reward(a,b,c)=({args.reward_a:.2f},{args.reward_b:.2f},{args.reward_c:.2f}), "
            f"reward_scale={args.reward_scale:.2f}, anti_repeat_l={args.anti_repeat_lambda:.2f}, "
            f"diversity_l={args.candidate_diversity_lambda:.2f}, "
            f"expected_hit=Y lambda={args.expected_hit_lambda:.2f}, synergy={args.expected_hit_synergy_lambda:.2f}, "
            f"latestN={args.latest_priority_n}, latestBoost={args.latest_priority_boost:.2f}, "
            f"latestL={args.latest_priority_lambda:.2f}, latestAddlL={args.latest_priority_addl_lambda:.2f}"
        )
    objective_hit_focused = bool(args.hit_score_focused and (not args.avg_hit_focused))
    if bool(args.avg_hit_focused) and bool(args.hit_score_focused):
        print(
            "[AVG-FOCUS] objective override: using avg-oriented tuning/restart/blend scoring "
            "(tail-hit objective disabled for model selection)"
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
    graph_prior_matrix = pipe["graph_prior_matrix"]
    embed_prior_matrix = pipe["embed_prior_matrix"]
    addl_embed_prior_matrix = pipe["addl_embed_prior_matrix"]
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
    _mark_stage("features_and_sequences")

    # Hyperparameter tuning via walk-forward validation.
    trial_configs = generate_trial_configs(args.tune_trials)
    recent_mode = args.focus_last_n > 0
    print(
        f"Running tuning: trials={len(trial_configs)}, folds={args.tune_folds}, "
        f"epochs={args.tune_epochs}, recent_focus={recent_mode}, last_n={args.focus_last_n}"
    )
    tuning_df = run_tuning_multifidelity(
        trial_configs=trial_configs,
        seq_cache=seq_cache,
        df=df,
        win_cluster_prior_matrix=win_cluster_prior_matrix,
        graph_prior_matrix=graph_prior_matrix,
        embed_prior_matrix=embed_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=BLEND_WEIGHTS,
        tune_folds=args.tune_folds,
        tune_epochs=args.tune_epochs,
        focus_last_n=args.focus_last_n,
        batch_size=int(hw["batch_size"]),
        hit_score_focused=objective_hit_focused,
        steps_per_execution=int(args.steps_per_execution),
        cache_dataset=bool(args.dataset_cache),
        train_recency_weighted=bool(args.train_recency_weighted),
        dynamic_blend=bool(args.uncertainty_dynamic_blend),
        enable_multifidelity=bool(args.tune_multifidelity),
        stage1_epochs=int(args.tune_mf_stage1_epochs),
        keep_ratio=float(args.tune_mf_keep_ratio),
        candidate_max_pool=int(args.candidate_max_pool),
        candidate_gumbel_samples=int(args.candidate_gumbel_samples),
        candidate_diversity_lambda=float(args.candidate_diversity_lambda),
        expected_hit_rerank=bool(args.expected_hit_rerank),
        expected_hit_lambda=float(args.expected_hit_lambda),
        expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
        anti_repeat_window=int(args.anti_repeat_window),
        anti_repeat_lambda=float(args.anti_repeat_lambda),
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
    _mark_stage("tuning")

    # Final training using chronological train/val/test split.
    X_seq, y_raw, target_rows = seq_cache[best_config.seq_len]
    train_idx, val_idx, test_idx = split_train_val_test(len(X_seq))
    if args.focus_last_n > 0:
        restart_eval_idx = recent_seq_indices(len(X_seq), args.focus_last_n, min_train=max(120, best_config.seq_len * 4))
        if len(restart_eval_idx) == 0:
            restart_eval_idx = val_idx
        restart_eval_weights = recency_weights_for_target_rows(df, target_rows[np.asarray(restart_eval_idx, dtype=np.int32)])
        restart_eval_weights = emphasize_latest_weights(
            restart_eval_weights,
            latest_n=int(args.latest_priority_n),
            boost=float(args.latest_priority_boost),
            power=1.9,
        )
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
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=BLEND_WEIGHTS,
            sample_weights=restart_eval_weights,
            dynamic_blend=bool(args.uncertainty_dynamic_blend),
            candidate_max_pool=int(args.candidate_max_pool),
            candidate_gumbel_samples=int(args.candidate_gumbel_samples),
            candidate_diversity_lambda=float(args.candidate_diversity_lambda),
            expected_hit_rerank=bool(args.expected_hit_rerank),
            expected_hit_lambda=float(args.expected_hit_lambda),
            expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
            anti_repeat_window=int(args.anti_repeat_window),
            anti_repeat_lambda=float(args.anti_repeat_lambda),
        )
        restart_score = float(blend_objective(restart_metrics, focused=objective_hit_focused))
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
    _mark_stage("final_training_and_test")

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
        blend_eval_weights = emphasize_latest_weights(
            blend_eval_weights,
            latest_n=int(args.latest_priority_n),
            boost=float(args.latest_priority_boost),
            power=1.9,
        )
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
        graph_prior_matrix=graph_prior_matrix,
        embed_prior_matrix=embed_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        base_weights=BLEND_WEIGHTS,
        sample_weights=blend_eval_weights,
        cached_preds=blend_eval_preds,
        dynamic_blend=bool(args.uncertainty_dynamic_blend),
        candidate_max_pool=int(args.candidate_max_pool),
        candidate_gumbel_samples=int(args.candidate_gumbel_samples),
        candidate_diversity_lambda=float(args.candidate_diversity_lambda),
        expected_hit_rerank=bool(args.expected_hit_rerank),
        expected_hit_lambda=float(args.expected_hit_lambda),
        expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
        anti_repeat_window=int(args.anti_repeat_window),
        anti_repeat_lambda=float(args.anti_repeat_lambda),
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
    blend_focused = objective_hit_focused
    for cand in blend_candidates:
        val_hit_metrics = evaluate_hit_score(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=blend_eval_idx,
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=cand,
            sample_weights=blend_eval_weights,
            cached_preds=blend_eval_preds,
            dynamic_blend=bool(args.uncertainty_dynamic_blend),
            candidate_max_pool=int(args.candidate_max_pool),
            candidate_gumbel_samples=int(args.candidate_gumbel_samples),
            candidate_diversity_lambda=float(args.candidate_diversity_lambda),
            expected_hit_rerank=bool(args.expected_hit_rerank),
            expected_hit_lambda=float(args.expected_hit_lambda),
            expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
            anti_repeat_window=int(args.anti_repeat_window),
            anti_repeat_lambda=float(args.anti_repeat_lambda),
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
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            addl_cluster_prior_matrix=addl_cluster_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            weights=cand,
            sample_weights=blend_eval_weights,
            cached_preds=blend_eval_preds,
            dynamic_blend=bool(args.uncertainty_dynamic_blend),
            candidate_max_pool=int(args.candidate_max_pool),
            candidate_gumbel_samples=int(args.candidate_gumbel_samples),
            candidate_diversity_lambda=float(args.candidate_diversity_lambda),
            expected_hit_rerank=bool(args.expected_hit_rerank),
            expected_hit_lambda=float(args.expected_hit_lambda),
            expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
            anti_repeat_window=int(args.anti_repeat_window),
            anti_repeat_lambda=float(args.anti_repeat_lambda),
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

    reward_reranker: RewardReranker | None = None
    if bool(args.reward_rerank):
        reward_reranker = fit_reward_reranker(
            model=model,
            scaler_x=scaler_x,
            X_seq=X_seq,
            eval_idx=np.asarray(blend_eval_idx, dtype=np.int32),
            target_rows=target_rows,
            df=df,
            win_cluster_prior_matrix=win_cluster_prior_matrix,
            graph_prior_matrix=graph_prior_matrix,
            embed_prior_matrix=embed_prior_matrix,
            repel_prior_matrix=repel_prior_matrix,
            blend_weights=selected_blend,
            dynamic_blend=bool(args.uncertainty_dynamic_blend),
            reward_a=float(args.reward_a),
            reward_b=float(args.reward_b),
            reward_c=float(args.reward_c),
            reward_window=int(args.reward_window),
            hardneg_boost=float(args.reward_hardneg_boost),
            ridge=float(args.reward_ridge),
            reward_scale=float(args.reward_scale),
            candidate_max_pool=int(args.candidate_max_pool),
            candidate_gumbel_samples=int(args.candidate_gumbel_samples),
            candidate_diversity_lambda=float(args.candidate_diversity_lambda),
        )
        if reward_reranker is None:
            print("[REWARD] reranker disabled (insufficient training samples).")
        else:
            print(
                "[REWARD] reranker fitted: "
                f"window={args.reward_window}, a={args.reward_a:.2f}, b={args.reward_b:.2f}, "
                f"c={args.reward_c:.2f}, scale={args.reward_scale:.2f}"
            )
    _mark_stage("blend_search")

    # Walk-forward expanding-window backtest for historical validation.
    if args.focus_last_n > 0:
        mode_note = "no-retrain" if bool(args.backtest_no_retrain) else "retrain-per-step"
        print(
            f"Running strict walk-forward backtest on last {args.focus_last_n} draws, "
            f"epochs={args.backtest_epochs}, mode={mode_note}"
        )
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
        graph_prior_matrix=graph_prior_matrix,
        embed_prior_matrix=embed_prior_matrix,
        addl_cluster_prior_matrix=addl_cluster_prior_matrix,
        repel_prior_matrix=repel_prior_matrix,
        blend_weights=selected_blend,
        backtest_folds=args.backtest_folds,
        backtest_epochs=args.backtest_epochs,
        focus_last_n=args.focus_last_n,
        latest_priority_n=int(args.latest_priority_n),
        latest_priority_boost=float(args.latest_priority_boost),
        latest_priority_lambda=float(args.latest_priority_lambda),
        latest_priority_addl_lambda=float(args.latest_priority_addl_lambda),
        batch_size=int(hw["batch_size"]),
        local_random_candidates=int(args.backtest_local_random_candidates),
        optimize_avg=bool(args.backtest_optimize_avg),
        backtest_restarts=int(args.backtest_restarts),
        restart_ensemble_topk=int(args.restart_ensemble_topk),
        steps_per_execution=int(args.steps_per_execution),
        cache_dataset=bool(args.dataset_cache),
        train_recency_weighted=bool(args.train_recency_weighted),
        dynamic_blend=bool(args.uncertainty_dynamic_blend),
        reward_reranker=reward_reranker,
        candidate_max_pool=int(args.candidate_max_pool),
        candidate_gumbel_samples=int(args.candidate_gumbel_samples),
        candidate_diversity_lambda=float(args.candidate_diversity_lambda),
        expected_hit_rerank=bool(args.expected_hit_rerank),
        expected_hit_lambda=float(args.expected_hit_lambda),
        expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
        anti_repeat_window=int(args.anti_repeat_window),
        anti_repeat_lambda=float(args.anti_repeat_lambda),
        no_retrain=bool(args.backtest_no_retrain),
        inference_model=model,
        inference_scaler=scaler_x,
    )
    _mark_stage("backtest")

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
    embed_next_prior = normalize_prob(embed_prior_matrix[-1])

    latest_seq = scaler_x.transform(X_seq[-1].reshape(-1, n_features)).reshape(1, X_seq.shape[1], n_features).astype(np.float32)
    pred_latest = predict_outputs_dict(model, latest_seq)
    pred_win, pred_addl = combine_prediction(
        pred=pred_latest,
        win_cluster_prior=win_cluster_next_prior,
        graph_prior=normalize_prob(graph_prior_matrix[-1]),
        embed_prior=embed_next_prior,
        repel_prior=repel_next_prior,
        weights=selected_blend,
        dynamic_blend=bool(args.uncertainty_dynamic_blend),
        reward_reranker=reward_reranker,
        candidate_max_pool=int(args.candidate_max_pool),
        candidate_gumbel_samples=int(args.candidate_gumbel_samples),
        candidate_diversity_lambda=float(args.candidate_diversity_lambda),
        expected_hit_rerank=bool(args.expected_hit_rerank),
        expected_hit_lambda=float(args.expected_hit_lambda),
        expected_hit_synergy_lambda=float(args.expected_hit_synergy_lambda),
        anti_repeat_window=int(args.anti_repeat_window),
        anti_repeat_lambda=float(args.anti_repeat_lambda),
    )

    pred_stats = {
        "hybrid_win_numbers": pred_win,
        "hybrid_addl_number": pred_addl,
        "cluster_last": int(cluster_last),
        "cluster_next": int(cluster_next),
        "pred_sum": int(sum(pred_win)),
        "pred_mean": float(np.mean(pred_win)),
        "low_count": int(sum(1 for n in pred_win if n <= 24)),
        "high_count": int(sum(1 for n in pred_win if n > 24)),
        "odd_count": int(sum(1 for n in pred_win if n % 2 == 1)),
        "even_count": int(sum(1 for n in pred_win if n % 2 == 0)),
    }

    # Interactive dashboard payload (Grafana-style Plotly charts).
    history_loss = [float(x) for x in final_history.history.get("loss", [])]
    history_val_loss = [float(x) for x in final_history.history.get("val_loss", [])]
    history_epoch = list(range(1, len(history_loss) + 1))

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
        transition_matrix=transition_matrix,
        dashboard=dashboard,
        focus_last_n=args.focus_last_n,
        feature_group_weights=feature_group_weights,
        feature_weight_tuning_df=feature_weight_tuning_df,
        blend_component_df=blend_component_df,
        calibrated_blend=calibrated_blend,
    )
    _mark_stage("report_write")

    print("")
    print("Prediction (latest):")
    print(f"Win: {pred_win}")
    print(f"Addl: {pred_addl}")
    print("")
    print("Backtest summary:")
    print(backtest_summary)
    if not backtest_df.empty and "pred_win" in backtest_df.columns:
        rep = backtest_df["pred_win"].value_counts()
        rep = rep[rep > 1]
        if not rep.empty:
            top_rep = rep.head(8).to_dict()
            print(f"[BACKTEST] repeated pred_win sets detected: {top_rep}")
    print("")
    print("Stage runtime (seconds):")
    for k, v in stage_times.items():
        print(f"  {k}: {v:.2f}")
    print(f"  total: {sum(stage_times.values()):.2f}")
    print("")
    print(f"Single HTML report generated: {html_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()

