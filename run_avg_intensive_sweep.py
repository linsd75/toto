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
class AvgConfig:
    name: str
    seed: int
    multi_restarts: int
    tail_iters: int
    tail_cands: int
    simplex_iters: int
    simplex_step: float
    random_cands: int
    local_random: int
    final_epochs: int
    backtest_epochs: int
    tune_trials: int
    tune_epochs: int
    coord_iters: int
    coord_step: float
    reward_epochs: int
    reward_window: int
    focus_last_n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avg-win-hits intensive sweep anchored on selected report families.")
    parser.add_argument("--csv", default="ToTo-05_Mar_2026.csv", help="Input CSV")
    parser.add_argument("--train-script", default="Predict_Toto_HTML.py", help="Training script path")
    parser.add_argument("--target-avg", type=float, default=2.5, help="Early-stop threshold for avg_win_hits")
    parser.add_argument("--max-runs", type=int, default=0, help="Cap number of runs (0 = all)")
    return parser.parse_args()


def parse_backtest_summary(stdout: str) -> Dict[str, float]:
    marker = "Backtest summary:"
    idx = stdout.rfind(marker)
    if idx < 0:
        return {}
    tail = stdout[idx + len(marker) :].strip().splitlines()
    for line in tail:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                raw = ast.literal_eval(line)
                return {str(k): float(v) for k, v in raw.items()}
            except Exception:
                return {}
    return {}


def parse_output_html(stdout: str) -> str:
    marker = "Single HTML report generated:"
    idx = stdout.rfind(marker)
    if idx < 0:
        return ""
    return stdout[idx + len(marker) :].strip().splitlines()[0].strip()


def objective(row: Dict[str, float]) -> float:
    avg = float(row.get("avg_win_hits", 0.0))
    wavg = float(row.get("weighted_avg_win_hits", 0.0))
    p3 = float(row.get("p_hit_ge_3", 0.0))
    p4 = float(row.get("p_hit_ge_4", 0.0))
    wp3 = float(row.get("weighted_p_hit_ge_3", 0.0))
    wp4 = float(row.get("weighted_p_hit_ge_4", 0.0))
    addl = float(row.get("weighted_addl_acc", 0.0))
    bonus = 0.0
    if avg >= 1.8:
        bonus += 120.0
    if avg >= 2.0:
        bonus += 240.0
    if avg >= 2.5:
        bonus += 520.0
    if avg >= 3.0:
        bonus += 820.0
    return (
        780.0 * avg
        + 300.0 * wavg
        + 135.0 * p3
        + 220.0 * p4
        + 95.0 * wp3
        + 170.0 * wp4
        + 18.0 * addl
        + bonus
    )


def build_configs() -> List[AvgConfig]:
    # Anchors from user-provided report families.
    # report_01_s42_A.html
    a = AvgConfig("s42_A_base", 42, 5, 8, 360, 160, 0.24, 3600, 2600, 64, 20, 12, 10, 14, 0.42, 10, 320, 10)
    # report_06_s236_B.html
    b = AvgConfig("s236_B_base", 236, 6, 10, 520, 220, 0.28, 5200, 4200, 80, 24, 14, 12, 14, 0.42, 10, 320, 10)
    # report_02_s42_r4_t8c300_s120.html
    c = AvgConfig("s42_r4_t8c300_s120_base", 42, 4, 8, 300, 120, 0.24, 2200, 2200, 42, 12, 8, 8, 8, 0.34, 6, 220, 10)
    # report_04_s139_r4_t8c300_s120.html
    d = AvgConfig("s139_r4_t8c300_s120_base", 139, 4, 8, 300, 120, 0.24, 2200, 2200, 42, 12, 8, 8, 8, 0.34, 6, 220, 10)
    # report_05_s236_r3_t6c200_s96.html
    e = AvgConfig("s236_r3_t6c200_s96_base", 236, 3, 6, 200, 96, 0.20, 1700, 1700, 42, 12, 8, 8, 8, 0.34, 6, 220, 10)

    cfgs: List[AvgConfig] = [a, b, c, d, e]

    # Aggressive local variants for avg-hit push.
    cfgs.extend(
        [
            AvgConfig("s42_A_v1", 42, 6, 10, 520, 220, 0.28, 5200, 4200, 80, 24, 14, 12, 16, 0.48, 12, 398, 12),
            AvgConfig("s42_A_v2", 42, 7, 10, 640, 260, 0.30, 6200, 5200, 92, 26, 16, 12, 18, 0.50, 13, 420, 12),
            AvgConfig("s236_B_v1", 236, 7, 12, 680, 280, 0.30, 7200, 6200, 96, 28, 18, 13, 18, 0.50, 13, 420, 12),
            AvgConfig("s236_B_v2", 236, 8, 12, 760, 320, 0.32, 8200, 7200, 108, 30, 20, 14, 20, 0.54, 14, 460, 12),
            AvgConfig("s42_r4_v1", 42, 5, 8, 360, 160, 0.24, 3600, 2800, 64, 18, 12, 10, 14, 0.42, 10, 300, 10),
            AvgConfig("s139_r4_v1", 139, 5, 8, 360, 160, 0.24, 3600, 2800, 64, 18, 12, 10, 14, 0.42, 10, 300, 10),
            AvgConfig("s236_r3_v1", 236, 4, 8, 300, 120, 0.24, 2800, 2400, 56, 16, 10, 9, 12, 0.40, 8, 260, 10),
        ]
    )
    return cfgs


def run_one(
    root: Path,
    py: Path,
    train_py: Path,
    csv: str,
    out_dir: Path,
    idx: int,
    cfg: AvgConfig,
) -> Dict[str, float]:
    html_rel = f"{out_dir.as_posix()}/report_{idx:02d}_{cfg.name}.html"
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
        "320",
        "--diffusion-batch-size",
        "128",
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
    print(f"[{idx}] {cfg.name} ...", flush=True)
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path = out_dir / f"log_{idx:02d}_{cfg.name}.txt"
    log_path.write_text(proc.stdout + "\n\n[STDERR]\n" + proc.stderr, encoding="utf-8")

    summ = parse_backtest_summary(proc.stdout)
    out_html = parse_output_html(proc.stdout)
    row: Dict[str, float] = {
        "run": idx,
        "name": cfg.name,
        "status_code": float(proc.returncode),
        "html_path": out_html if out_html else html_rel,
        "log_path": str(log_path),
        "seed": float(cfg.seed),
        "multi_restarts": float(cfg.multi_restarts),
        "tail_iters": float(cfg.tail_iters),
        "tail_cands": float(cfg.tail_cands),
        "simplex_iters": float(cfg.simplex_iters),
        "simplex_step": float(cfg.simplex_step),
        "random_cands": float(cfg.random_cands),
        "local_random": float(cfg.local_random),
        "final_epochs": float(cfg.final_epochs),
        "backtest_epochs": float(cfg.backtest_epochs),
        "tune_trials": float(cfg.tune_trials),
        "tune_epochs": float(cfg.tune_epochs),
        "coord_iters": float(cfg.coord_iters),
        "coord_step": float(cfg.coord_step),
        "reward_epochs": float(cfg.reward_epochs),
        "reward_window": float(cfg.reward_window),
        "focus_last_n": float(cfg.focus_last_n),
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "models" / f"avg_intensive_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    if int(args.max_runs) > 0:
        configs = configs[: int(args.max_runs)]

    rows: List[Dict[str, float]] = []
    best_avg = -1e18
    for i, cfg in enumerate(configs, start=1):
        row = run_one(root=root, py=py, train_py=train_py, csv=str(args.csv), out_dir=out_dir, idx=i, cfg=cfg)
        rows.append(row)
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

    out_csv = out_dir / "avg_intensive_results.csv"
    out_json = out_dir / "avg_intensive_summary.json"
    df.to_csv(out_csv, index=False)
    payload = {
        "timestamp": ts,
        "csv": str(args.csv),
        "train_script": str(train_py),
        "target_avg": float(args.target_avg),
        "results_csv": str(out_csv),
        "best": best,
        "runs_requested": len(configs),
        "runs_completed": len(rows),
        "base_families": [
            "report_01_s42_A",
            "report_06_s236_B",
            "report_02_s42_r4_t8c300_s120",
            "report_04_s139_r4_t8c300_s120",
            "report_05_s236_r3_t6c200_s96",
        ],
        "configs": [asdict(c) for c in configs],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nAvg-intensive sweep complete.")
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
