from __future__ import annotations

import argparse
import ast
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class Sweep16Config:
    name: str
    seed: int
    multi_restarts: int
    tune_trials: int
    tune_epochs: int
    final_epochs: int
    backtest_epochs: int
    random_cands: int
    coord_iters: int
    coord_step: float
    simplex_iters: int
    simplex_step: float
    tail_iters: int
    tail_cands: int
    local_random: int
    reward_epochs: int
    reward_window: int
    focus_last_n: int
    gpu_batch_size: int
    backtest_restarts: int
    restart_topk: int
    train_recency_weighted: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable 16-run avg-win-hits sweep for Predict_Toto_HTML.py")
    parser.add_argument("--csv", default="ToTo-05_Mar_2026.csv", help="Input CSV")
    parser.add_argument("--train-script", default="Predict_Toto_HTML.py", help="Training script path")
    parser.add_argument("--target-avg", type=float, default=99.0, help="Early-stop threshold for avg_win_hits")
    parser.add_argument("--max-runs", type=int, default=16, help="Cap number of configs executed")
    parser.add_argument("--resume-dir", default="", help="Resume into an existing models/sweep_* directory")
    return parser.parse_args()


def parse_backtest_summary(text: str) -> Dict[str, float]:
    marker = "Backtest summary:"
    idx = text.rfind(marker)
    if idx < 0:
        return {}
    tail = text[idx + len(marker) :].strip().splitlines()
    for line in tail:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                raw = ast.literal_eval(line)
                return {str(k): float(v) for k, v in raw.items()}
            except Exception:
                return {}
    return {}


def parse_output_html(text: str) -> str:
    marker = "Single HTML report generated:"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    return text[idx + len(marker) :].strip().splitlines()[0].strip()


def objective(row: Dict[str, float]) -> float:
    avg = float(row.get("avg_win_hits", 0.0))
    wavg = float(row.get("weighted_avg_win_hits", 0.0))
    p3 = float(row.get("p_hit_ge_3", 0.0))
    p4 = float(row.get("p_hit_ge_4", 0.0))
    wp3 = float(row.get("weighted_p_hit_ge_3", 0.0))
    wp4 = float(row.get("weighted_p_hit_ge_4", 0.0))
    addl = float(row.get("weighted_addl_acc", 0.0))
    return (
        760.0 * avg
        + 320.0 * wavg
        + 150.0 * p3
        + 230.0 * p4
        + 95.0 * wp3
        + 180.0 * wp4
        + 20.0 * addl
    )


def build_configs() -> List[Sweep16Config]:
    cfgs: List[Sweep16Config] = []
    seeds = [42, 139, 236, 333]
    # A: stable moderate baseline.
    base_a = dict(
        multi_restarts=3,
        tune_trials=8,
        tune_epochs=8,
        final_epochs=50,
        backtest_epochs=14,
        random_cands=2200,
        coord_iters=10,
        coord_step=0.38,
        simplex_iters=96,
        simplex_step=0.22,
        tail_iters=6,
        tail_cands=220,
        local_random=1400,
        reward_epochs=8,
        reward_window=260,
        focus_last_n=10,
        gpu_batch_size=320,
        backtest_restarts=1,
        restart_topk=1,
        train_recency_weighted=False,
    )
    # B: moderate + recency focus.
    base_b = dict(
        multi_restarts=4,
        tune_trials=10,
        tune_epochs=9,
        final_epochs=58,
        backtest_epochs=16,
        random_cands=2800,
        coord_iters=12,
        coord_step=0.40,
        simplex_iters=120,
        simplex_step=0.24,
        tail_iters=7,
        tail_cands=280,
        local_random=1800,
        reward_epochs=9,
        reward_window=300,
        focus_last_n=10,
        gpu_batch_size=320,
        backtest_restarts=1,
        restart_topk=1,
        train_recency_weighted=True,
    )
    # C: stronger v2-like.
    base_c = dict(
        multi_restarts=5,
        tune_trials=12,
        tune_epochs=10,
        final_epochs=64,
        backtest_epochs=18,
        random_cands=3400,
        coord_iters=14,
        coord_step=0.42,
        simplex_iters=150,
        simplex_step=0.26,
        tail_iters=8,
        tail_cands=360,
        local_random=2400,
        reward_epochs=10,
        reward_window=320,
        focus_last_n=12,
        gpu_batch_size=352,
        backtest_restarts=1,
        restart_topk=1,
        train_recency_weighted=True,
    )
    # D: ensemble-in-backtest variant.
    base_d = dict(
        multi_restarts=5,
        tune_trials=12,
        tune_epochs=10,
        final_epochs=64,
        backtest_epochs=18,
        random_cands=3400,
        coord_iters=14,
        coord_step=0.42,
        simplex_iters=150,
        simplex_step=0.26,
        tail_iters=8,
        tail_cands=360,
        local_random=2400,
        reward_epochs=10,
        reward_window=320,
        focus_last_n=12,
        gpu_batch_size=352,
        backtest_restarts=2,
        restart_topk=2,
        train_recency_weighted=True,
    )

    variants = [("A", base_a), ("B", base_b), ("C", base_c), ("D", base_d)]
    for seed in seeds:
        for label, base in variants:
            cfgs.append(
                Sweep16Config(
                    name=f"s{seed}_{label}",
                    seed=seed,
                    **base,
                )
            )
    return cfgs


def build_row_from_log(cfg: Sweep16Config, idx: int, log_text: str, fallback_html: str, log_path: Path) -> Dict[str, float]:
    summ = parse_backtest_summary(log_text)
    out_html = parse_output_html(log_text)
    row: Dict[str, float] = {
        "run": idx,
        "name": cfg.name,
        "status_code": 0.0 if "Single HTML report generated:" in log_text else 1.0,
        "html_path": out_html if out_html else fallback_html,
        "log_path": str(log_path),
        "seed": float(cfg.seed),
        "multi_restarts": float(cfg.multi_restarts),
        "tune_trials": float(cfg.tune_trials),
        "tune_epochs": float(cfg.tune_epochs),
        "final_epochs": float(cfg.final_epochs),
        "backtest_epochs": float(cfg.backtest_epochs),
        "random_cands": float(cfg.random_cands),
        "coord_iters": float(cfg.coord_iters),
        "coord_step": float(cfg.coord_step),
        "simplex_iters": float(cfg.simplex_iters),
        "simplex_step": float(cfg.simplex_step),
        "tail_iters": float(cfg.tail_iters),
        "tail_cands": float(cfg.tail_cands),
        "local_random": float(cfg.local_random),
        "reward_epochs": float(cfg.reward_epochs),
        "reward_window": float(cfg.reward_window),
        "focus_last_n": float(cfg.focus_last_n),
        "gpu_batch_size": float(cfg.gpu_batch_size),
        "backtest_restarts": float(cfg.backtest_restarts),
        "restart_topk": float(cfg.restart_topk),
        "train_recency_weighted": float(1 if cfg.train_recency_weighted else 0),
    }
    for k in [
        "avg_win_hits",
        "p_hit_ge_2",
        "p_hit_ge_3",
        "p_hit_ge_4",
        "p_exact6",
        "addl_acc",
        "weighted_avg_win_hits",
        "weighted_p_hit_ge_2",
        "weighted_p_hit_ge_3",
        "weighted_p_hit_ge_4",
        "weighted_p_exact6",
        "weighted_addl_acc",
    ]:
        row[k] = float(summ.get(k, 0.0))
    row["objective"] = float(objective(row))
    return row


def run_one(
    root: Path,
    py: Path,
    train_py: Path,
    csv: str,
    out_dir: Path,
    idx: int,
    cfg: Sweep16Config,
) -> Dict[str, float]:
    html_abs = out_dir / f"report_{idx:02d}_{cfg.name}.html"
    html_rel = str(html_abs)
    log_path = out_dir / f"log_{idx:02d}_{cfg.name}.txt"

    if html_abs.exists() and log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        row = build_row_from_log(cfg, idx, text, html_rel, log_path)
        print(
            f"[{idx}] {cfg.name} (resume) "
            f"avg={row['avg_win_hits']:.4f} p3={row['p_hit_ge_3']:.4f} p4={row['p_hit_ge_4']:.4f}",
            flush=True,
        )
        return row

    cmd = [
        str(py),
        str(train_py),
        "--csv",
        str(csv),
        "--seed",
        str(cfg.seed),
        "--hit-score-focused",
        "--sweep-mode",
        "--backtest-optimize-avg",
        "--focus-last-n",
        str(cfg.focus_last_n),
        "--multi-restarts",
        str(cfg.multi_restarts),
        "--gpu-preallocate",
        "--gpu-batch-size",
        str(cfg.gpu_batch_size),
        "--diffusion-batch-size",
        "128",
        "--perf-mode",
        "high",
        "--steps-per-execution",
        "32",
        "--dataset-cache",
        "--tune-trials",
        str(cfg.tune_trials),
        "--tune-epochs",
        str(cfg.tune_epochs),
        "--final-epochs",
        str(cfg.final_epochs),
        "--backtest-folds",
        "10",
        "--backtest-epochs",
        str(cfg.backtest_epochs),
        "--blend-random-candidates",
        str(cfg.random_cands),
        "--blend-coordinate-iters",
        str(cfg.coord_iters),
        "--blend-coordinate-step",
        str(cfg.coord_step),
        "--blend-simplex-iters",
        str(cfg.simplex_iters),
        "--blend-simplex-step",
        str(cfg.simplex_step),
        "--blend-tail-iters",
        str(cfg.tail_iters),
        "--blend-tail-candidates",
        str(cfg.tail_cands),
        "--backtest-local-random-candidates",
        str(cfg.local_random),
        "--backtest-restarts",
        str(cfg.backtest_restarts),
        "--restart-ensemble-topk",
        str(cfg.restart_topk),
        "--diffusion-trials",
        "1",
        "--diffusion-epochs",
        "4",
        "--diffusion-steps",
        "28",
        "--diffusion-samples",
        "3",
        "--diffusion-future-samples",
        "64",
        "--diffusion-window",
        "64",
        "--reward-epochs",
        str(cfg.reward_epochs),
        "--reward-window",
        str(cfg.reward_window),
        "--output-html",
        html_rel,
    ]
    if cfg.train_recency_weighted:
        cmd.append("--train-recency-weighted")

    print(f"[{idx}] {cfg.name} ...", flush=True)
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path.write_text(proc.stdout + "\n\n[STDERR]\n" + proc.stderr, encoding="utf-8")
    row = build_row_from_log(cfg, idx, proc.stdout, html_rel, log_path)
    row["status_code"] = float(proc.returncode)
    print(
        f"    exit={proc.returncode} avg={row['avg_win_hits']:.4f} "
        f"w_avg={row['weighted_avg_win_hits']:.4f} p3={row['p_hit_ge_3']:.4f} "
        f"p4={row['p_hit_ge_4']:.4f} obj={row['objective']:.2f}",
        flush=True,
    )
    return row


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = root / ".venv_tf_cuda" / "Scripts" / "python.exe"
    train_py = root / str(args.train_script)
    if not py.exists():
        raise FileNotFoundError(f"Python not found: {py}")
    if not train_py.exists():
        raise FileNotFoundError(f"Training script not found: {train_py}")

    if str(args.resume_dir).strip():
        out_dir = Path(args.resume_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "models" / f"avg_sweep16_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    max_runs = max(1, min(int(args.max_runs), len(configs)))
    configs = configs[:max_runs]

    rows: List[Dict[str, float]] = []
    best_avg = -1e18
    interim_csv = out_dir / "sweep16_results_interim.csv"

    for i, cfg in enumerate(configs, start=1):
        row = run_one(root=root, py=py, train_py=train_py, csv=str(args.csv), out_dir=out_dir, idx=i, cfg=cfg)
        rows.append(row)
        pd.DataFrame(rows).to_csv(interim_csv, index=False)
        best_avg = max(best_avg, float(row.get("avg_win_hits", 0.0)))
        if best_avg >= float(args.target_avg):
            print(
                f"[EARLY-STOP] reached target avg_win_hits={best_avg:.4f} "
                f"(target={float(args.target_avg):.4f})",
                flush=True,
            )
            break

    df = pd.DataFrame(rows)
    ok = df[df["status_code"] == 0.0].copy() if not df.empty else pd.DataFrame()
    if not ok.empty:
        ok = ok.sort_values(
            ["objective", "avg_win_hits", "weighted_avg_win_hits", "p_hit_ge_4", "p_hit_ge_3"],
            ascending=False,
        ).reset_index(drop=True)
        best = ok.iloc[0].to_dict()
    else:
        best = {}

    out_csv = out_dir / "sweep16_results.csv"
    out_json = out_dir / "sweep16_summary.json"
    df.to_csv(out_csv, index=False)
    payload = {
        "csv": str(args.csv),
        "train_script": str(train_py),
        "target_avg": float(args.target_avg),
        "results_csv": str(out_csv),
        "best": best,
        "runs_requested": len(configs),
        "runs_completed": len(rows),
        "search_mode": "16-run moderate/aggressive mix with checkpointing",
        "configs": [asdict(c) for c in configs],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSweep16 complete.")
    print(f"Results CSV: {out_csv}")
    print(f"Summary: {out_json}")
    if best:
        print(
            f"Best: {best.get('name')} avg={float(best.get('avg_win_hits', 0.0)):.4f} "
            f"w_avg={float(best.get('weighted_avg_win_hits', 0.0)):.4f} "
            f"p3={float(best.get('p_hit_ge_3', 0.0)):.4f} "
            f"p4={float(best.get('p_hit_ge_4', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()

