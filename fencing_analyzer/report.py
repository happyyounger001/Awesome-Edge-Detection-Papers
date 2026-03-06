from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

from .config import AnalyzerConfig
from .metrics import LungeMetrics, aggregate_summary, metric_flags


def _save_plots(fig_dir: Path, metrics: List[LungeMetrics]) -> Dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)

    durations = [m.duration for m in metrics]
    plt.figure(figsize=(6, 3))
    plt.bar(range(1, len(durations) + 1), durations)
    plt.title("Lunge Duration by Attempt")
    plt.xlabel("Lunge ID")
    plt.ylabel("Duration (s)")
    dur_name = "duration.png"
    plt.tight_layout()
    plt.savefig(fig_dir / dur_name)
    plt.close()

    plt.figure(figsize=(6, 3))
    knee = [m.knee_angle_min for m in metrics]
    trunk = [m.trunk_lean_max for m in metrics]
    plt.plot(range(1, len(knee) + 1), knee, marker="o", label="knee min")
    plt.plot(range(1, len(trunk) + 1), trunk, marker="x", label="trunk max")
    plt.title("Posture Metrics")
    plt.xlabel("Lunge ID")
    plt.ylabel("Degrees")
    plt.legend()
    post_name = "posture_metrics.png"
    plt.tight_layout()
    plt.savefig(fig_dir / post_name)
    plt.close()

    rt = [m.reaction_time for m in metrics if m.reaction_time is not None]
    plt.figure(figsize=(6, 3))
    if rt:
        plt.hist(rt, bins=min(6, len(rt)))
        plt.title("Reaction Time Distribution")
        plt.xlabel("Seconds")
    else:
        plt.text(0.5, 0.5, "Reaction time unavailable", ha="center", va="center")
        plt.axis("off")
    rt_name = "reaction_time.png"
    plt.tight_layout()
    plt.savefig(fig_dir / rt_name)
    plt.close()

    order = Counter(m.foot_hand_order for m in metrics)
    plt.figure(figsize=(5, 3))
    if order:
        plt.bar(order.keys(), order.values())
        plt.ylabel("Count")
    plt.title("Foot-Hand Order")
    fho_name = "foot_hand_order.png"
    plt.tight_layout()
    plt.savefig(fig_dir / fho_name)
    plt.close()

    return {
        "duration": f"figures/{dur_name}",
        "posture": f"figures/{post_name}",
        "reaction": f"figures/{rt_name}",
        "order": f"figures/{fho_name}",
    }


def write_summary_json(out_path: Path, metrics: List[LungeMetrics]) -> Dict[str, float]:
    summary = aggregate_summary(metrics)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_report(
    output_dir: Path,
    video_name: str,
    metrics: List[LungeMetrics],
    cfg: AnalyzerConfig,
    beep_enabled: bool,
) -> Path:
    fig_paths = _save_plots(output_dir / "figures", metrics)
    summary = write_summary_json(output_dir / "summary.json", metrics)

    env = Environment(loader=FileSystemLoader(str(Path(__file__).resolve().parent.parent / "templates")))
    template = env.get_template("report.html.j2")

    rows = []
    for m in metrics:
        rows.append({
            **m.__dict__,
            **metric_flags(m, cfg),
        })

    html = template.render(
        video_name=video_name,
        summary=summary,
        rows=rows,
        fig_paths=fig_paths,
        beep_enabled=beep_enabled,
    )
    out_file = output_dir / "report.html"
    out_file.write_text(html, encoding="utf-8")
    return out_file
