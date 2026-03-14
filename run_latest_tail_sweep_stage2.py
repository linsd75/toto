from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class TailSweepConfig:
    label: str
    seed: int
    reward_a: float
    reward_b: float
    reward_c: float
    anti: float
    div: float
    eh: float
    esy: float
    w_graph: float
    w_cluster: float
    latest_boost: float
    latest_lambda: float
    latest_addl_lambda: float


def parse_metric(html_path: Path, metric_name: str) -> float:
    if not html_path.exists():
        return float("nan")
    txt = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(rf"<strong>{re.escape(metric_name)}</strong><br>([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse_backtest_rows(html_path: Path) -> List[Dict[str, str]]:
    if not html_path.exists():
        return []
    txt = html_path.read_text(encoding="utf-8", errors="ignore")
    pos = txt.find("Backtest Detail Table")
    if pos < 0:
        return []
    t0 = txt.find("<table", pos)
    t1 = txt.find("</table>", t0)
    if t0 < 0 or t1 < 0:
        return []
    table_html = txt[t0 : t1 + len("</table>")]
    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
    if not rows_html:
        return []

    headers: List[str] = []
    for r in rows_html:
        ths = re.findall(r"<th[^>]*>(.*?)</th>", r, flags=re.IGNORECASE | re.DOTALL)
        if ths:
            cand = [_strip_tags(x) for x in ths]
            if "win_hits" in cand and "addl_hit" in cand:
                headers = cand
                break
    if not headers:
        return []

    out: List[Dict[str, str]] = []
    for r in rows_html:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.IGNORECASE | re.DOTALL)
        if not tds:
            continue
        vals = [_strip_tags(x) for x in tds]
        if len(vals) != len(headers):
            continue
        rec = {headers[i]: vals[i] for i in range(len(headers))}
        if "win_hits" in rec and "addl_hit" in rec:
            out.append(rec)
    return out


def tail_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {
            "last5_avg_hits": float("nan"),
            "last5_addl_acc": float("nan"),
            "last5_ge3_rate": float("nan"),
            "last3_avg_hits": float("nan"),
            "last3_ge4_count": float("nan"),
            "last3_min_hits": float("nan"),
        }

    vals = []
    for r in rows:
        try:
            h = float(r.get("win_hits", "nan"))
            a = float(r.get("addl_hit", "nan"))
        except Exception:
            continue
        if h == h and a == a:
            vals.append((h, a))
    if not vals:
        return {
            "last5_avg_hits": float("nan"),
            "last5_addl_acc": float("nan"),
            "last5_ge3_rate": float("nan"),
            "last3_avg_hits": float("nan"),
            "last3_ge4_count": float("nan"),
            "last3_min_hits": float("nan"),
        }

    l5 = vals[-min(5, len(vals)) :]
    l3 = vals[-min(3, len(vals)) :]
    l5_hits = [x[0] for x in l5]
    l5_addl = [x[1] for x in l5]
    l3_hits = [x[0] for x in l3]
    return {
        "last5_avg_hits": float(sum(l5_hits) / len(l5_hits)),
        "last5_addl_acc": float(sum(l5_addl) / len(l5_addl)),
        "last5_ge3_rate": float(sum(1.0 for h in l5_hits if h >= 3.0) / len(l5_hits)),
        "last3_avg_hits": float(sum(l3_hits) / len(l3_hits)),
        "last3_ge4_count": float(sum(1.0 for h in l3_hits if h >= 4.0)),
        "last3_min_hits": float(min(l3_hits)),
    }


def score_row(row: Dict[str, float]) -> float:
    def z(v: float, default: float = -1e9) -> float:
        return default if v != v else float(v)

    return (
        460.0 * z(row["last3_ge4_count"])
        + 180.0 * z(row["last3_avg_hits"])
        + 90.0 * z(row["last3_min_hits"])
        + 76.0 * z(row["last5_avg_hits"])
        + 36.0 * z(row["last5_ge3_rate"])
        + 22.0 * z(row["last5_addl_acc"])
        + 3.0 * z(row["avg_win_hits"])
        + 1.0 * z(row["p_hit_ge_3_pct"])
        + 0.8 * z(row["p_hit_ge_4_pct"])
    )


def write_summaries(out_dir: Path, rows: List[Dict[str, float]], best: Dict[str, float] | None) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=score_row, reverse=True)
    payload = {"best": best, "results": rows_sorted}
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Second-stage local tail sweep around best profile")
    ap.add_argument("--csv", default="ToTo-12_Mar_2026.csv", help="Input CSV path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    py = root / ".venv_tf_cuda" / "Scripts" / "python.exe"
    train_py = root / "Predict_Toto_HTML.py"
    out_dir = root / "models" / f"latest_tail_sweep_stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage-2 neighborhood around T02_tailhard.
    configs: List[TailSweepConfig] = [
        TailSweepConfig("S2_A_anchor", 139, 2.05, 2.35, 5.20, 0.45, 0.07, 3.20, 0.86, 2.05, 0.95, 5.2, 3.8, 3.0),
        TailSweepConfig("S2_B_tail_up", 237, 2.10, 2.45, 5.80, 0.40, 0.06, 3.45, 0.94, 2.15, 0.90, 5.9, 4.4, 3.4),
        TailSweepConfig("S2_C_tail_up2", 333, 2.12, 2.52, 6.20, 0.35, 0.06, 3.60, 1.02, 2.25, 0.86, 6.4, 5.0, 3.8),
        TailSweepConfig("S2_D_addl_push", 430, 2.00, 2.38, 5.40, 0.42, 0.06, 3.30, 0.90, 2.10, 0.92, 5.6, 4.0, 4.2),
        TailSweepConfig("S2_E_lowanti", 527, 2.08, 2.42, 5.60, 0.30, 0.07, 3.35, 0.92, 2.15, 0.90, 5.8, 4.6, 3.4),
        TailSweepConfig("S2_F_medanti", 624, 2.02, 2.30, 5.10, 0.52, 0.08, 3.05, 0.82, 2.00, 0.98, 4.9, 3.5, 2.8),
        TailSweepConfig("S2_G_cluster_bias", 721, 2.06, 2.36, 5.30, 0.45, 0.07, 3.25, 0.88, 1.95, 1.04, 5.4, 3.9, 3.0),
        TailSweepConfig("S2_H_graph_bias", 818, 2.06, 2.36, 5.30, 0.44, 0.06, 3.30, 0.90, 2.30, 0.86, 5.5, 4.2, 3.1),
    ]

    base_args = [
        str(py),
        str(train_py),
        "--csv",
        str(args.csv),
        "--avg-hit-focused",
        "--no-hit-score-focused",
        "--sweep-mode",
        "--focus-last-n",
        "24",
        "--latest-priority-n",
        "5",
        "--gpu-preallocate",
        "--gpu-batch-size",
        "512",
        "--steps-per-execution",
        "32",
        "--dataset-cache",
        "--train-recency-weighted",
        "--tune-trials",
        "5",
        "--tune-epochs",
        "3",
        "--tune-multifidelity",
        "--tune-mf-stage1-epochs",
        "2",
        "--tune-mf-keep-ratio",
        "0.5",
        "--multi-restarts",
        "2",
        "--final-epochs",
        "16",
        "--backtest-folds",
        "24",
        "--backtest-epochs",
        "8",
        "--backtest-no-retrain",
        "--blend-random-candidates",
        "1000",
        "--blend-coordinate-iters",
        "8",
        "--blend-coordinate-step",
        "0.30",
        "--blend-simplex-iters",
        "48",
        "--blend-simplex-step",
        "0.18",
        "--backtest-local-random-candidates",
        "1400",
        "--candidate-max-pool",
        "84",
        "--candidate-gumbel-samples",
        "64",
    ]

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "1"
    rows: List[Dict[str, float]] = []
    best: Dict[str, float] | None = None

    for i, cfg in enumerate(configs, start=1):
        html_path = out_dir / f"run_{i:02d}_{cfg.label}.html"
        log_path = out_dir / f"run_{i:02d}_{cfg.label}.log"
        cmd = base_args + [
            "--seed",
            str(cfg.seed),
            "--reward-a",
            f"{cfg.reward_a}",
            "--reward-b",
            f"{cfg.reward_b}",
            "--reward-c",
            f"{cfg.reward_c}",
            "--anti-repeat-lambda",
            f"{cfg.anti}",
            "--candidate-diversity-lambda",
            f"{cfg.div}",
            "--expected-hit-lambda",
            f"{cfg.eh}",
            "--expected-hit-synergy-lambda",
            f"{cfg.esy}",
            "--w-graph",
            f"{cfg.w_graph}",
            "--w-cluster",
            f"{cfg.w_cluster}",
            "--latest-priority-boost",
            f"{cfg.latest_boost}",
            "--latest-priority-lambda",
            f"{cfg.latest_lambda}",
            "--latest-priority-addl-lambda",
            f"{cfg.latest_addl_lambda}",
            "--output-html",
            str(html_path),
        ]
        print(f"[RUN {i}/{len(configs)}] {cfg.label} seed={cfg.seed}")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        log_path.write_text((p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8", errors="ignore")

        avg = parse_metric(html_path, "Avg Win Hits")
        p3 = parse_metric(html_path, "P(Hits >= 3)")
        p4 = parse_metric(html_path, "P(Hits >= 4)")
        tail = tail_metrics(parse_backtest_rows(html_path))
        row: Dict[str, float] = {
            "run": f"run_{i:02d}_{cfg.label}",
            "seed": cfg.seed,
            "avg_win_hits": avg,
            "p_hit_ge_3_pct": p3,
            "p_hit_ge_4_pct": p4,
            "last5_avg_hits": tail["last5_avg_hits"],
            "last5_addl_acc": tail["last5_addl_acc"],
            "last5_ge3_rate": tail["last5_ge3_rate"],
            "last3_avg_hits": tail["last3_avg_hits"],
            "last3_ge4_count": tail["last3_ge4_count"],
            "last3_min_hits": tail["last3_min_hits"],
            "exit_code": p.returncode,
            "html": str(html_path.relative_to(root)),
            "log": str(log_path.relative_to(root)),
            "latest_boost": cfg.latest_boost,
            "latest_lambda": cfg.latest_lambda,
            "latest_addl_lambda": cfg.latest_addl_lambda,
            "reward_a": cfg.reward_a,
            "reward_b": cfg.reward_b,
            "reward_c": cfg.reward_c,
            "anti": cfg.anti,
            "div": cfg.div,
            "eh": cfg.eh,
            "esy": cfg.esy,
            "w_graph": cfg.w_graph,
            "w_cluster": cfg.w_cluster,
        }
        row["objective"] = score_row(row)
        rows.append(row)
        print(
            "  -> "
            f"last3_ge4={row['last3_ge4_count']:.0f} "
            f"last3_avg={row['last3_avg_hits']:.3f} "
            f"last5_avg={row['last5_avg_hits']:.3f} "
            f"avg={row['avg_win_hits']:.3f} "
            f"obj={row['objective']:.2f} "
            f"exit={p.returncode}"
        )
        if best is None or score_row(row) > score_row(best):
            best = row
        write_summaries(out_dir, rows, best)
        if row["last3_ge4_count"] == 3:
            print(f"[EARLY-STOP] ideal latest-3>=4 achieved by {row['run']}")
            break

    write_summaries(out_dir, rows, best)
    print(f"\nOutput folder: {out_dir}")
    if best is not None:
        print(
            f"Best: {best['run']} "
            f"last3_ge4={best['last3_ge4_count']:.0f} "
            f"last3_avg={best['last3_avg_hits']:.4f} "
            f"last5_avg={best['last5_avg_hits']:.4f} "
            f"avg={best['avg_win_hits']:.4f} "
            f"objective={best['objective']:.2f}"
        )


if __name__ == "__main__":
    main()

