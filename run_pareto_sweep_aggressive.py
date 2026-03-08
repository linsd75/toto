from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class AggressiveConfig:
    name: str
    seed: int
    multi_restarts: int
    tail_iters: int
    tail_cands: int
    simplex_iters: int
    simplex_step: float
    random_cands: int
    local_random: int
    optimize_avg: bool
    hit_focus: bool
    final_epochs: int
    backtest_epochs: int
    tune_trials: int
    tune_epochs: int


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
    line = stdout[idx + len(marker) :].strip().splitlines()[0].strip()
    return line


def dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    ge_all = True
    gt_one = False
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        if bv < av:
            ge_all = False
            break
        if bv > av:
            gt_one = True
    return ge_all and gt_one


def pareto_front(rows: List[Dict[str, float]], keys: List[str]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for r in rows:
        if any(dominates(r, q, keys) for q in rows if q is not r):
            continue
        out.append(r)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    train_py = root / "train_toto_tuned_single_html.py"
    py = root / ".venv_tf_cuda" / "Scripts" / "python.exe"
    if not py.exists():
        raise FileNotFoundError(f"Python not found: {py}")
    if not train_py.exists():
        raise FileNotFoundError(f"Training script not found: {train_py}")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "models" / f"sweep_aggressive_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Designed around your good runs (report_02/04/05), but with more aggressive local blend search.
    configs: List[AggressiveConfig] = [
        AggressiveConfig("s42_A", 42, 5, 8, 360, 160, 0.24, 3600, 2600, True, True, 64, 20, 12, 10),
        AggressiveConfig("s42_B", 42, 6, 10, 520, 220, 0.28, 5200, 4200, True, True, 80, 24, 14, 12),
        AggressiveConfig("s139_A", 139, 5, 8, 360, 160, 0.24, 3600, 2600, True, True, 64, 20, 12, 10),
        AggressiveConfig("s139_B", 139, 6, 10, 520, 220, 0.28, 5200, 4200, True, True, 80, 24, 14, 12),
        AggressiveConfig("s236_A", 236, 5, 8, 360, 160, 0.24, 3600, 2600, True, True, 64, 20, 12, 10),
        AggressiveConfig("s236_B", 236, 6, 10, 520, 220, 0.28, 5200, 4200, True, True, 80, 24, 14, 12),
    ]

    rows: List[Dict[str, float]] = []
    for i, cfg in enumerate(configs, start=1):
        html_rel = f"models/sweep_aggressive_{now}/report_{i:02d}_{cfg.name}.html"
        cmd = [
            str(py),
            str(train_py),
            "--csv",
            "ToTo-05_Mar_2026.csv",
            "--seed",
            str(cfg.seed),
            "--focus-last-n",
            "10",
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
            "--backtest-epochs",
            str(cfg.backtest_epochs),
            "--blend-random-candidates",
            str(cfg.random_cands),
            "--blend-coordinate-iters",
            "14",
            "--blend-coordinate-step",
            "0.42",
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
            "10",
            "--reward-window",
            "320",
            "--output-html",
            html_rel,
        ]
        if cfg.hit_focus:
            cmd.append("--hit-score-focused")
        if cfg.optimize_avg:
            cmd.append("--backtest-optimize-avg")

        print(f"[{i}/{len(configs)}] {cfg.name} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
        log_path = out_dir / f"log_{i:02d}_{cfg.name}.txt"
        log_path.write_text(proc.stdout + "\n\n[STDERR]\n" + proc.stderr, encoding="utf-8")
        summ = parse_backtest_summary(proc.stdout)
        out_html = parse_output_html(proc.stdout)

        row: Dict[str, float] = {
            "run": i,
            "name": cfg.name,
            "seed": cfg.seed,
            "status_code": float(proc.returncode),
            "html_path": out_html if out_html else html_rel,
            "log_path": str(log_path),
            "multi_restarts": float(cfg.multi_restarts),
            "tail_iters": float(cfg.tail_iters),
            "tail_cands": float(cfg.tail_cands),
            "simplex_iters": float(cfg.simplex_iters),
            "simplex_step": float(cfg.simplex_step),
            "random_cands": float(cfg.random_cands),
            "local_random": float(cfg.local_random),
            "optimize_avg": float(1 if cfg.optimize_avg else 0),
            "hit_focus": float(1 if cfg.hit_focus else 0),
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
        rows.append(row)
        print(
            f"    exit={proc.returncode} avg={row['avg_win_hits']:.4f} "
            f"w_avg={row['weighted_avg_win_hits']:.4f} p3={row['p_hit_ge_3']:.4f} p4={row['p_hit_ge_4']:.4f}",
            flush=True,
        )

    ok = [r for r in rows if int(r.get("status_code", 1)) == 0]
    pareto_tail = pareto_front(ok, keys=["weighted_p_hit_ge_3", "weighted_p_hit_ge_4"])
    pareto_avg_tail = pareto_front(ok, keys=["avg_win_hits", "p_hit_ge_3", "p_hit_ge_4"])

    best_avg = sorted(
        ok,
        key=lambda x: (
            float(x.get("avg_win_hits", 0.0)),
            float(x.get("weighted_avg_win_hits", 0.0)),
            float(x.get("p_hit_ge_3", 0.0)),
            float(x.get("p_hit_ge_4", 0.0)),
        ),
        reverse=True,
    )[0] if ok else {}
    best_weighted_avg = sorted(
        ok,
        key=lambda x: (
            float(x.get("weighted_avg_win_hits", 0.0)),
            float(x.get("avg_win_hits", 0.0)),
            float(x.get("weighted_p_hit_ge_3", 0.0)),
            float(x.get("weighted_p_hit_ge_4", 0.0)),
        ),
        reverse=True,
    )[0] if ok else {}

    df = pd.DataFrame(rows)
    out_csv = out_dir / "aggressive_sweep_results.csv"
    pareto_tail_csv = out_dir / "aggressive_pareto_tail.csv"
    pareto_avg_tail_csv = out_dir / "aggressive_pareto_avg_tail.csv"
    out_json = out_dir / "aggressive_summary.json"
    df.to_csv(out_csv, index=False)
    pd.DataFrame(pareto_tail).to_csv(pareto_tail_csv, index=False)
    pd.DataFrame(pareto_avg_tail).to_csv(pareto_avg_tail_csv, index=False)
    payload = {
        "timestamp": now,
        "results_csv": str(out_csv),
        "pareto_tail_csv": str(pareto_tail_csv),
        "pareto_avg_tail_csv": str(pareto_avg_tail_csv),
        "best_avg": best_avg,
        "best_weighted_avg": best_weighted_avg,
        "pareto_tail": pareto_tail,
        "pareto_avg_tail": pareto_avg_tail,
        "runs": [asdict(c) for c in configs],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nAggressive sweep complete.")
    print(f"Results: {out_csv}")
    print(f"Pareto (tail): {pareto_tail_csv}")
    print(f"Pareto (avg+tail): {pareto_avg_tail_csv}")
    print(f"Summary: {out_json}")
    if best_avg:
        print(
            f"Best avg_win_hits: {best_avg.get('name')} "
            f"avg={best_avg.get('avg_win_hits', 0.0):.4f} "
            f"p3={best_avg.get('p_hit_ge_3', 0.0):.4f} p4={best_avg.get('p_hit_ge_4', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
