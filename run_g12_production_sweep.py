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
from typing import Dict, List, TextIO


@dataclass
class G12Config:
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
    w_embed: float
    latest_boost: float
    latest_lambda: float
    latest_addl_lambda: float
    tune_trials: int
    tune_epochs: int
    final_epochs: int
    backtest_epochs: int
    blend_random: int
    blend_coord: int
    blend_simplex: int
    candidate_pool: int
    candidate_gumbel: int


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def detect_gpu_count() -> int:
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=8)
        if p.returncode != 0:
            return 0
        return sum(1 for line in (p.stdout or "").splitlines() if line.strip().startswith("GPU "))
    except Exception:
        return 0


def csv_draw_summary(csv_path: Path) -> tuple[int, int | None, int | None]:
    rows = 0
    draw_min: int | None = None
    draw_max: int | None = None
    if not csv_path.exists():
        return rows, draw_min, draw_max
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            d_raw = row.get("Draw", "")
            try:
                d = int(float(str(d_raw).strip()))
            except Exception:
                continue
            draw_min = d if draw_min is None else min(draw_min, d)
            draw_max = d if draw_max is None else max(draw_max, d)
    return rows, draw_min, draw_max


def run_streamed(
    cmd: List[str],
    env: Dict[str, str],
    run_log_path: Path,
    log_info,
    run_prefix: str,
) -> int:
    with run_log_path.open("w", encoding="utf-8", errors="ignore") as run_log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        if proc.stdout is not None:
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                run_log.write(line + "\n")
                run_log.flush()
                if line:
                    log_info(f"[{run_prefix}] {line}")
        return int(proc.wait())


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
        out.append({headers[i]: vals[i] for i in range(len(headers))})
    return out


def tail_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {"last5_avg_hits": float("nan"), "last5_ge3_rate": float("nan"), "last5_addl_acc": float("nan")}
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
        return {"last5_avg_hits": float("nan"), "last5_ge3_rate": float("nan"), "last5_addl_acc": float("nan")}
    l5 = vals[-min(5, len(vals)) :]
    l5_hits = [x[0] for x in l5]
    l5_addl = [x[1] for x in l5]
    return {
        "last5_avg_hits": float(sum(l5_hits) / len(l5_hits)),
        "last5_ge3_rate": float(sum(1.0 for h in l5_hits if h >= 3.0) / len(l5_hits)),
        "last5_addl_acc": float(sum(l5_addl) / len(l5_addl)),
    }


def score_row(row: Dict[str, float]) -> float:
    def z(v: float, default: float = -1e9) -> float:
        return default if v != v else float(v)

    last5_avg = z(row["last5_avg_hits"])
    tail_bonus = 0.0
    if last5_avg >= 4.0:
        tail_bonus += 4200.0 + 1200.0 * min(1.0, last5_avg - 4.0)
    return (
        420.0 * z(row["avg_win_hits"])
        + 180.0 * z(row["p_hit_ge_3_pct"])
        + 240.0 * z(row["p_hit_ge_4_pct"])
        + 960.0 * last5_avg
        + 420.0 * z(row["last5_ge3_rate"])
        + 80.0 * z(row["last5_addl_acc"])
        + tail_bonus
    )


def write_summary(out_dir: Path, rows: List[Dict[str, float]], best: Dict[str, float] | None) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=score_row, reverse=True)
    payload = {"best": best, "results": rows_sorted}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)


def build_cmd(
    py: Path,
    train_py: Path,
    csv_path: str,
    html_path: Path,
    cfg: G12Config,
) -> List[str]:
    return [
        str(py),
        "-u",
        str(train_py),
        "--csv",
        csv_path,
        "--seed",
        str(cfg.seed),
        "--output-html",
        str(html_path),
        "--avg-hit-focused",
        "--no-hit-score-focused",
        "--sweep-mode",
        "--focus-last-n",
        "24",
        "--process-priority",
        "high",
        "--latest-priority-n",
        "5",
        "--gpu-preallocate",
        "--gpu-batch-size",
        "640",
        "--steps-per-execution",
        "32",
        "--dataset-cache",
        "--tune-trials",
        str(cfg.tune_trials),
        "--tune-epochs",
        str(cfg.tune_epochs),
        "--tune-multifidelity",
        "--tune-mf-stage1-epochs",
        "2",
        "--tune-mf-keep-ratio",
        "0.5",
        "--multi-restarts",
        "3",
        "--final-epochs",
        str(cfg.final_epochs),
        "--backtest-folds",
        "24",
        "--backtest-epochs",
        str(cfg.backtest_epochs),
        "--backtest-no-retrain",
        "--backtest-restarts",
        "2",
        "--restart-ensemble-topk",
        "2",
        "--backtest-parallel-workers",
        "0",
        "--backtest-progress-every",
        "25",
        "--blend-nested-holdout",
        "--blend-holdout-size",
        "6",
        "--blend-adaptive-early-stop",
        "--blend-early-stop-min-evals",
        "320",
        "--blend-early-stop-patience",
        "220",
        "--backtest-adaptive-early-stop",
        "--backtest-early-stop-min-evals",
        "560",
        "--backtest-early-stop-patience",
        "360",
        "--blend-random-candidates",
        str(cfg.blend_random),
        "--blend-coordinate-iters",
        str(cfg.blend_coord),
        "--blend-simplex-iters",
        str(cfg.blend_simplex),
        "--candidate-max-pool",
        str(cfg.candidate_pool),
        "--candidate-gumbel-samples",
        str(cfg.candidate_gumbel),
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
        "--w-embed",
        f"{cfg.w_embed}",
        "--latest-priority-boost",
        f"{cfg.latest_boost}",
        "--latest-priority-lambda",
        f"{cfg.latest_lambda}",
        "--latest-priority-addl-lambda",
        f"{cfg.latest_addl_lambda}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Production sweep for Predict_Toto_G12.py")
    ap.add_argument("--csv", default="ToTo-12_Mar_2026.csv", help="Input CSV path")
    ap.add_argument("--runs", type=int, default=32, help="Max number of configs to run")
    ap.add_argument(
        "--final-upgrade",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upgrade winner into a heavier final pass (can improve or degrade; default keeps exact winner config)",
    )
    ap.add_argument(
        "--combined-log",
        default="models/g12_sweep_combined.log",
        help="Combined live log path (single file, timestamped info + streamed run output)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    py = root / ".venv_tf_cuda" / "Scripts" / "python.exe"
    train_py = root / "Predict_Toto_G12.py"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "models" / f"g12_prod_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_log_path = (root / args.combined_log).resolve()
    combined_log_path.parent.mkdir(parents=True, exist_ok=True)
    out_combined_log_path = out_dir / "sweep_combined.log"

    combined_files: List[TextIO] = []
    combined_files.append(combined_log_path.open("w", encoding="utf-8", errors="ignore"))
    combined_files.append(out_combined_log_path.open("w", encoding="utf-8", errors="ignore"))

    def log_info(msg: str) -> None:
        line = f"{_ts()} [INFO] {msg}"
        print(line, flush=True)
        for fh in combined_files:
            fh.write(line + "\n")
            fh.flush()

    # 32-run win-factor sweep from discovered ranges:
    # C-mid family (16 runs):
    # reward_a~1.85-1.98, reward_b~2.72-2.90, reward_c~7.40-8.05
    # N-high family (16 runs):
    # reward_a~2.06-2.18, reward_b~3.02-3.20, reward_c~8.40-8.90
    # Shared stable ranges:
    # w_cluster~=1.35, w_graph~1.40-1.54, anti_repeat~0.80-0.95, diversity~0.09-0.10,
    # expected_lambda~2.6-3.2, synergy~0.50-0.70, latestBoost~3.65-4.60, latestL~2.10-2.70
    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def iround(x: float) -> int:
        return int(round(x))

    configs: List[G12Config] = []
    tvals = [i / 15.0 for i in range(16)]

    for i, t in enumerate(tvals):
        # C-mid (lower-moderate pressure)
        configs.append(
            G12Config(
                f"C{i + 1:02d}",
                721 + i * 17,
                round(lerp(1.85, 1.98, t), 3),
                round(lerp(2.72, 2.90, t), 3),
                round(lerp(7.40, 8.05, t), 3),
                round(lerp(0.95, 0.82, t), 3),
                round(lerp(0.10, 0.09, t), 3),
                round(lerp(2.60, 3.00, t), 3),
                round(lerp(0.50, 0.62, t), 3),
                round(lerp(1.40, 1.48, t), 3),
                1.35,
                round(lerp(1.34, 1.50, t), 3),
                round(lerp(3.65, 4.30, t), 3),
                round(lerp(2.10, 2.45, t), 3),
                round(lerp(1.20, 1.55, t), 3),
                iround(lerp(6, 7, t)),
                iround(lerp(6, 7, t)),
                iround(lerp(24, 28, t)),
                iround(lerp(8, 10, t)),
                iround(lerp(1300, 1750, t)),
                iround(lerp(10, 12, t)),
                iround(lerp(80, 108, t)),
                iround(lerp(76, 94, t)),
                iround(lerp(48, 66, t)),
            )
        )

    for i, t in enumerate(tvals):
        # N-high (higher pressure)
        configs.append(
            G12Config(
                f"N{i + 1:02d}",
                1801 + i * 17,
                round(lerp(2.06, 2.18, t), 3),
                round(lerp(3.02, 3.20, t), 3),
                round(lerp(8.40, 8.90, t), 3),
                round(lerp(0.88, 0.80, t), 3),
                round(lerp(0.095, 0.09, t), 3),
                round(lerp(3.00, 3.20, t), 3),
                round(lerp(0.60, 0.70, t), 3),
                round(lerp(1.46, 1.54, t), 3),
                1.35,
                round(lerp(1.56, 1.70, t), 3),
                round(lerp(4.25, 4.60, t), 3),
                round(lerp(2.45, 2.70, t), 3),
                round(lerp(1.55, 1.80, t), 3),
                iround(lerp(7, 8, t)),
                iround(lerp(7, 8, t)),
                iround(lerp(28, 32, t)),
                iround(lerp(10, 12, t)),
                iround(lerp(1750, 2200, t)),
                iround(lerp(12, 14, t)),
                iround(lerp(104, 132, t)),
                iround(lerp(92, 108, t)),
                iround(lerp(66, 86, t)),
            )
        )

    try:
        to_run = configs[: max(1, min(int(args.runs), len(configs)))]
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "1"

        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = (root / csv_path).resolve()
        draw_rows, draw_min, draw_max = csv_draw_summary(csv_path)
        gpu_count = detect_gpu_count()

        log_info("Mixed precision float16 enabled.")
        log_info(f"GPU(s) detected: {gpu_count}")
        log_info("============================================================")
        log_info("Phase 1: Loading data")
        log_info("============================================================")
        if draw_min is not None and draw_max is not None:
            log_info(f"Loaded {draw_rows} draws  (Draw {draw_min} to {draw_max})")
        else:
            log_info(f"Loaded {draw_rows} draws")
        log_info("============================================================")
        log_info("Phase 2: Sweep runs")
        log_info("============================================================")
        log_info(f"Output folder: {out_dir}")
        log_info(f"Combined log: {combined_log_path}")

        rows: List[Dict[str, float]] = []
        best: Dict[str, float] | None = None

        for i, cfg in enumerate(to_run, start=1):
            html_path = out_dir / f"run_{i:02d}_{cfg.label}.html"
            log_path = out_dir / f"run_{i:02d}_{cfg.label}.log"
            cmd = build_cmd(py=py, train_py=train_py, csv_path=args.csv, html_path=html_path, cfg=cfg)
            log_info(f"[RUN {i}/{len(to_run)}] {cfg.label} seed={cfg.seed}")
            exit_code = run_streamed(
                cmd=cmd,
                env=env,
                run_log_path=log_path,
                log_info=log_info,
                run_prefix=f"RUN{i:02d}-{cfg.label}",
            )

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
                "last5_ge3_rate": tail["last5_ge3_rate"],
                "last5_addl_acc": tail["last5_addl_acc"],
                "exit_code": exit_code,
                "html": str(html_path.relative_to(root)),
                "log": str(log_path.relative_to(root)),
            }
            row["objective"] = score_row(row)
            rows.append(row)
            if best is None or score_row(row) > score_row(best):
                best = row

            log_info(
                "-> "
                f"avg={row['avg_win_hits']:.4f} "
                f"p3={row['p_hit_ge_3_pct']:.2f} "
                f"p4={row['p_hit_ge_4_pct']:.2f} "
                f"last5_avg={row['last5_avg_hits']:.4f} "
                f"obj={row['objective']:.2f} "
                f"exit={row['exit_code']}"
            )
            write_summary(out_dir, rows, best)

        write_summary(out_dir, rows, best)
        if best is None:
            log_info("No successful run.")
            log_info(f"Output folder: {out_dir}")
            return

        best_run_name = str(best["run"])
        best_idx = int(best_run_name.split("_")[1]) - 1
        best_cfg = to_run[max(0, min(best_idx, len(to_run) - 1))]

        log_info("============================================================")
        log_info("Phase 3: Final Production Run From Sweep Winner")
        log_info("============================================================")

        final_html = out_dir / "best_final_production.html"
        final_log = out_dir / "best_final_production.log"
        final_cmd = build_cmd(py=py, train_py=train_py, csv_path=args.csv, html_path=final_html, cfg=best_cfg)
        # Optional heavier pass. Default is exact winner replay for stability/reproducibility.
        if bool(args.final_upgrade):
            final_cmd += [
                "--tune-trials",
                str(max(10, best_cfg.tune_trials + 2)),
                "--tune-epochs",
                str(max(10, best_cfg.tune_epochs + 4)),
                "--multi-restarts",
                "5",
                "--final-epochs",
                str(max(72, best_cfg.final_epochs * 2)),
                "--backtest-epochs",
                str(max(20, best_cfg.backtest_epochs + 8)),
                "--blend-random-candidates",
                str(max(3200, best_cfg.blend_random + 1200)),
                "--blend-coordinate-iters",
                str(max(16, best_cfg.blend_coord + 4)),
                "--blend-simplex-iters",
                str(max(180, best_cfg.blend_simplex + 60)),
                "--candidate-max-pool",
                str(max(128, best_cfg.candidate_pool + 28)),
                "--candidate-gumbel-samples",
                str(max(96, best_cfg.candidate_gumbel + 24)),
                "--output-html",
                str(final_html),
            ]

        log_info(f"[FINAL] Running production pass from winner: {best_run_name}")
        final_exit_code = run_streamed(
            cmd=final_cmd,
            env=env,
            run_log_path=final_log,
            log_info=log_info,
            run_prefix="FINAL",
        )

        final_avg = parse_metric(final_html, "Avg Win Hits")
        final_p3 = parse_metric(final_html, "P(Hits >= 3)")
        final_p4 = parse_metric(final_html, "P(Hits >= 4)")

        package = {
            "sweep_dir": str(out_dir.relative_to(root)),
            "best_run": best,
            "best_cfg": best_cfg.__dict__,
            "final_upgrade": bool(args.final_upgrade),
            "final_html": str(final_html.relative_to(root)),
            "final_log": str(final_log.relative_to(root)),
            "final_exit_code": final_exit_code,
            "final_metrics": {"avg_win_hits": final_avg, "p_hit_ge3_pct": final_p3, "p_hit_ge4_pct": final_p4},
            "final_command": final_cmd,
            "combined_log": str(combined_log_path),
            "combined_log_copy": str(out_combined_log_path.relative_to(root)),
        }
        (out_dir / "best_model_package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")

        log_info("============================================================")
        log_info("Completed")
        log_info("============================================================")
        log_info(f"Output folder: {out_dir}")
        log_info(f"Best sweep run: {best_run_name}")
        log_info(
            f"Final production metrics: avg={final_avg:.4f} "
            f"p3={final_p3:.2f} p4={final_p4:.2f} exit={final_exit_code}"
        )
    finally:
        for fh in combined_files:
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
