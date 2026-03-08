from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class SweepConfig:
    name: str
    seed: int
    multi_restarts: int
    tail_iters: int
    tail_cands: int
    simplex_iters: int
    simplex_step: float
    random_cands: int


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


def dominated(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return (
        float(b.get("weighted_p_hit_ge_4", 0.0)) >= float(a.get("weighted_p_hit_ge_4", 0.0))
        and float(b.get("weighted_p_hit_ge_3", 0.0)) >= float(a.get("weighted_p_hit_ge_3", 0.0))
        and (
            float(b.get("weighted_p_hit_ge_4", 0.0)) > float(a.get("weighted_p_hit_ge_4", 0.0))
            or float(b.get("weighted_p_hit_ge_3", 0.0)) > float(a.get("weighted_p_hit_ge_3", 0.0))
        )
    )


def pareto_front(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for r in rows:
        if any(dominated(r, q) for q in rows if q is not r):
            continue
        out.append(r)
    out.sort(
        key=lambda x: (
            float(x.get("weighted_p_hit_ge_4", 0.0)),
            float(x.get("weighted_p_hit_ge_3", 0.0)),
            float(x.get("weighted_avg_win_hits", 0.0)),
            float(x.get("p_hit_ge_4", 0.0)),
            float(x.get("p_hit_ge_3", 0.0)),
        ),
        reverse=True,
    )
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "train_toto_tuned_single_html.py"
    py = root / ".venv_tf_cuda" / "Scripts" / "python.exe"
    if not py.exists():
        raise FileNotFoundError(f"Python interpreter not found: {py}")
    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "models" / f"sweep_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: List[SweepConfig] = [
        SweepConfig("s42_r3_t6c200_s96", 42, 3, 6, 200, 96, 0.20, 1700),
        SweepConfig("s42_r4_t8c300_s120", 42, 4, 8, 300, 120, 0.24, 2200),
        SweepConfig("s139_r3_t6c200_s96", 139, 3, 6, 200, 96, 0.20, 1700),
        SweepConfig("s139_r4_t8c300_s120", 139, 4, 8, 300, 120, 0.24, 2200),
        SweepConfig("s236_r3_t6c200_s96", 236, 3, 6, 200, 96, 0.20, 1700),
        SweepConfig("s236_r4_t8c300_s120", 236, 4, 8, 300, 120, 0.24, 2200),
        SweepConfig("s333_r3_t6c200_s96", 333, 3, 6, 200, 96, 0.20, 1700),
        SweepConfig("s333_r4_t8c300_s120", 333, 4, 8, 300, 120, 0.24, 2200),
    ]

    rows: List[Dict[str, float]] = []
    for i, cfg in enumerate(configs, start=1):
        html_rel = f"models/sweep_{now}/report_{i:02d}_{cfg.name}.html"
        cmd = [
            str(py),
            str(script),
            "--csv",
            "ToTo-05_Mar_2026.csv",
            "--seed",
            str(cfg.seed),
            "--hit-score-focused",
            "--sweep-mode",
            "--focus-last-n",
            "10",
            "--multi-restarts",
            str(cfg.multi_restarts),
            "--gpu-preallocate",
            "--gpu-batch-size",
            "320",
            "--diffusion-batch-size",
            "160",
            "--tune-trials",
            "8",
            "--tune-epochs",
            "8",
            "--final-epochs",
            "42",
            "--backtest-epochs",
            "12",
            "--blend-random-candidates",
            str(cfg.random_cands),
            "--blend-coordinate-iters",
            "8",
            "--blend-coordinate-step",
            "0.34",
            "--blend-simplex-iters",
            str(cfg.simplex_iters),
            "--blend-simplex-step",
            str(cfg.simplex_step),
            "--blend-tail-iters",
            str(cfg.tail_iters),
            "--blend-tail-candidates",
            str(cfg.tail_cands),
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
            "6",
            "--reward-window",
            "220",
            "--output-html",
            html_rel,
        ]
        print(f"[{i}/{len(configs)}] Running {cfg.name} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
        log_path = out_dir / f"log_{i:02d}_{cfg.name}.txt"
        log_path.write_text(proc.stdout + "\n\n[STDERR]\n" + proc.stderr, encoding="utf-8")

        summary = parse_backtest_summary(proc.stdout)
        out_html = parse_output_html(proc.stdout)
        row: Dict[str, float] = {
            "run": i,
            "name": cfg.name,
            "seed": cfg.seed,
            "multi_restarts": cfg.multi_restarts,
            "tail_iters": cfg.tail_iters,
            "tail_cands": cfg.tail_cands,
            "simplex_iters": cfg.simplex_iters,
            "simplex_step": cfg.simplex_step,
            "random_cands": cfg.random_cands,
            "status_code": float(proc.returncode),
            "html_path": out_html if out_html else html_rel,
            "log_path": str(log_path),
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
            row[k] = float(summary.get(k, 0.0))
        rows.append(row)
        print(
            f"    exit={proc.returncode} "
            f"w_p3={row['weighted_p_hit_ge_3']:.4f} "
            f"w_p4={row['weighted_p_hit_ge_4']:.4f} "
            f"avg={row['avg_win_hits']:.4f}",
            flush=True,
        )

    ok_rows = [r for r in rows if int(r["status_code"]) == 0]
    front = pareto_front(ok_rows)
    best_p4 = sorted(
        ok_rows,
        key=lambda x: (
            float(x.get("weighted_p_hit_ge_4", 0.0)),
            float(x.get("weighted_p_hit_ge_3", 0.0)),
            float(x.get("weighted_avg_win_hits", 0.0)),
            float(x.get("p_hit_ge_4", 0.0)),
        ),
        reverse=True,
    )[0] if ok_rows else {}
    best_p3 = sorted(
        ok_rows,
        key=lambda x: (
            float(x.get("weighted_p_hit_ge_3", 0.0)),
            float(x.get("weighted_p_hit_ge_4", 0.0)),
            float(x.get("weighted_avg_win_hits", 0.0)),
            float(x.get("p_hit_ge_3", 0.0)),
        ),
        reverse=True,
    )[0] if ok_rows else {}

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sweep_results.csv"
    json_path = out_dir / "sweep_results.json"
    pareto_path = out_dir / "pareto_front.csv"
    df.to_csv(csv_path, index=False)
    pd.DataFrame(front).to_csv(pareto_path, index=False)

    payload = {
        "timestamp": now,
        "results_csv": str(csv_path),
        "pareto_csv": str(pareto_path),
        "best_weighted_p4": best_p4,
        "best_weighted_p3": best_p3,
        "pareto_front": front,
        "runs": [asdict(c) for c in configs],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSweep completed.")
    print(f"Results CSV: {csv_path}")
    print(f"Pareto CSV:  {pareto_path}")
    print(f"JSON:        {json_path}")
    if best_p4:
        print(
            f"Best weighted P(hit>=4): {best_p4.get('name')} "
            f"w_p4={best_p4.get('weighted_p_hit_ge_4', 0.0):.4f} "
            f"w_p3={best_p4.get('weighted_p_hit_ge_3', 0.0):.4f}"
        )
    if best_p3:
        print(
            f"Best weighted P(hit>=3): {best_p3.get('name')} "
            f"w_p3={best_p3.get('weighted_p_hit_ge_3', 0.0):.4f} "
            f"w_p4={best_p3.get('weighted_p_hit_ge_4', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
