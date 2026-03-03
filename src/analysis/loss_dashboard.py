#!/usr/bin/env python
"""
loss_dashboard.py

Live Gradio dashboard that monitors loss curves from multiple pretraining runs.

Each run stores its loss in:
    <checkpoints_root>/<run_name>/loss_log.csv

The dashboard auto-refreshes every few seconds so you can watch training in
real time, even with several runs in parallel.

Usage:
    python -m src.analysis.loss_dashboard          # default: watches ./checkpoints
    python -m src.analysis.loss_dashboard --dir ./checkpoints --port 7860
"""

import argparse
import re
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.configs.global_config import GLOBAL_CONFIG

CHECKPOINTS_ROOT = GLOBAL_CONFIG.SAVE_DIR  # default fallback
LOGS_DIR = GLOBAL_CONFIG.PROJECT_ROOT / "logs"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_csv_log(csv_path: Path) -> pd.DataFrame:
    """Read a loss_log.csv produced by train.py."""
    try:
        df = pd.read_csv(csv_path)
        df["epoch"] = df["epoch"].astype(int)
        df["loss"] = df["loss"].astype(float)
        if "lr" in df.columns:
            df["lr"] = df["lr"].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


def _parse_txt_log(txt_path: Path) -> pd.DataFrame:
    """
    Fallback: parse the raw stdout log files (output.txt, output_128.txt)
    that were produced before CSV logging was added.
    Format:  [Epoch 1/500] Loss: 3.6647  LR: 0.000300   (LR optional)
    """
    pattern = re.compile(
        r"\[Epoch\s+(\d+)/\d+]\s+Loss:\s+([\d.]+)(?:\s+LR:\s+([\d.]+))?"
    )
    rows = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    epoch = int(m.group(1))
                    loss = float(m.group(2))
                    lr = float(m.group(3)) if m.group(3) else None
                    rows.append({"epoch": epoch, "loss": loss, "lr": lr})
    except Exception:
        pass
    return pd.DataFrame(rows)


def _map_txt_to_run_name() -> dict:
    """
    Heuristic: map existing log text files to run names.
    output.txt       -> simclr_pretrain
    output_128.txt   -> simclr_pretrain_128
    output_<tag>.txt -> simclr_pretrain_<tag>
    """
    mapping = {}
    if not LOGS_DIR.exists():
        return mapping
    for txt in LOGS_DIR.glob("output*.txt"):
        stem = txt.stem  # e.g. "output" or "output_128"
        if stem == "output":
            run_name = "simclr_pretrain"
        else:
            suffix = stem.replace("output_", "")
            run_name = f"simclr_pretrain_{suffix}"
        mapping[run_name] = txt
    return mapping


def collect_all_runs(checkpoints_root: Path) -> dict:
    """
    Return {run_name: DataFrame} for every run found.
    Priority: loss_log.csv inside checkpoint dir > fallback txt log.
    """
    runs: dict[str, pd.DataFrame] = {}

    # 1) Scan checkpoint sub-directories for loss_log.csv
    if checkpoints_root.exists():
        for sub in sorted(checkpoints_root.iterdir()):
            if not sub.is_dir():
                continue
            csv_log = sub / "loss_log.csv"
            if csv_log.exists():
                df = _parse_csv_log(csv_log)
                if not df.empty:
                    runs[sub.name] = df

    # 2) Fallback: legacy text logs (only if CSV not already found)
    txt_map = _map_txt_to_run_name()
    for run_name, txt_path in txt_map.items():
        if run_name not in runs:
            df = _parse_txt_log(txt_path)
            if not df.empty:
                runs[run_name] = df

    return runs


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def build_loss_figure(checkpoints_root: Path):
    """Build a Plotly figure with one trace per run."""
    runs = collect_all_runs(checkpoints_root)

    fig = go.Figure()

    if not runs:
        fig.add_annotation(
            text="No training runs found yet. Start training!",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=18),
        )
        fig.update_layout(title="Loss Curves — waiting for data …")
        return fig

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    ]

    for i, (name, df) in enumerate(sorted(runs.items())):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["loss"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate="Epoch %{x}<br>Loss %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="SimCLR Pretraining — Live Loss Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Run",
        template="plotly_dark",
        hovermode="x unified",
        height=550,
    )

    return fig


def build_lr_figure(checkpoints_root: Path):
    """Build a Plotly figure showing learning-rate schedules."""
    runs = collect_all_runs(checkpoints_root)
    fig = go.Figure()

    if not runs:
        fig.update_layout(title="Learning Rate — waiting for data …")
        return fig

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    ]

    for i, (name, df) in enumerate(sorted(runs.items())):
        if "lr" not in df.columns or df["lr"].isna().all():
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["lr"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        legend_title="Run",
        template="plotly_dark",
        hovermode="x unified",
        height=350,
    )
    return fig


def build_summary_table(checkpoints_root: Path) -> pd.DataFrame:
    """Return a summary dataframe for all runs."""
    runs = collect_all_runs(checkpoints_root)
    rows = []
    for name, df in sorted(runs.items()):
        rows.append({
            "Run": name,
            "Epochs": int(df["epoch"].max()),
            "Latest Loss": f"{df['loss'].iloc[-1]:.4f}",
            "Min Loss": f"{df['loss'].min():.4f}",
            "Min @ Epoch": int(df.loc[df["loss"].idxmin(), "epoch"]),
            "Δ (first→last)": f"{df['loss'].iloc[-1] - df['loss'].iloc[0]:.4f}",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        [{"Run": "—", "Epochs": "—", "Latest Loss": "—",
          "Min Loss": "—", "Min @ Epoch": "—", "Δ (first→last)": "—"}]
    )


# ------------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------------

def create_dashboard(checkpoints_root: Path, refresh_interval: int = 10):
    """Create and return the Gradio Blocks app."""

    def refresh():
        return (
            build_loss_figure(checkpoints_root),
            build_lr_figure(checkpoints_root),
            build_summary_table(checkpoints_root),
        )

    with gr.Blocks(
        title="SimCLR Loss Dashboard",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as app:
        gr.Markdown("# 📉 SimCLR Pretraining — Loss Dashboard")
        gr.Markdown(
            f"Monitoring checkpoint directory: `{checkpoints_root}`\n\n"
            f"Auto-refreshes every **{refresh_interval}s**. "
            "You can also click **Refresh** manually."
        )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="primary", scale=0)

        loss_plot = gr.Plot(label="Loss Curves")
        lr_plot = gr.Plot(label="Learning Rate")
        summary_tbl = gr.Dataframe(label="Run Summary", interactive=False)

        # Initial load
        app.load(fn=refresh, outputs=[loss_plot, lr_plot, summary_tbl])

        # Manual refresh
        refresh_btn.click(fn=refresh, outputs=[loss_plot, lr_plot, summary_tbl])

        # Auto-refresh timer
        timer = gr.Timer(value=refresh_interval)
        timer.tick(fn=refresh, outputs=[loss_plot, lr_plot, summary_tbl])

    return app


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimCLR Loss Dashboard (Gradio)")
    parser.add_argument(
        "--dir", type=str, default=str(CHECKPOINTS_ROOT),
        help="Root directory containing checkpoint sub-folders",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True,
                        help="Create a public Gradio link (default: True)")
    parser.add_argument("--no-share", action="store_false", dest="share",
                        help="Disable public Gradio link")
    parser.add_argument("--refresh", type=int, default=10,
                        help="Auto-refresh interval in seconds")
    args = parser.parse_args()

    ckpt_root = Path(args.dir)
    print(f"📂 Watching: {ckpt_root.resolve()}")

    app = create_dashboard(ckpt_root, refresh_interval=args.refresh)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

