"""Stacked bars of how each target answered the 1-7 self-rating question, per style and
repetition group. Reads the 5-rep files and writes artifacts/<name>.png and .svg.

    uv run python experiments/welfare-interview-repro/scripts/plot_rating_modes.py \
        data/fullpass_5rep_haiku.jsonl data/fullpass_5rep_llama.jsonl data/fullpass_5rep_qwen.jsonl \
        --name rating_modes_5rep
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
LABELS = {"haiku45": "Haiku 4.5", "llama33-70b": "Llama 3.3 70B", "qwen3-30b-instruct": "Qwen3-30B", "opus55": "Opus 5.5"}
# Modes grouped into what a reader needs: gave a usable number, gave a range, refused for a reason, other.
GROUPS = [("number", ["number_first", "single_number", "range_midpoint"], "#2a78d6"),
          ("range", ["range"], "#8fb8ea"),
          ("refused: can't quantify", ["refusal_quantification"], "#eb6834"),
          ("refused: rejects frame", ["refusal_frame"], "#f3a27e"),
          ("refused: redirects", ["refusal_redirect"], "#c9c3bb"),
          ("no number / other", ["no_number", "other", "invalid_turn", "unparsed", None], "#8c8c8c")]
INK, INK2 = "#1f1f1f", "#6b6b6b"


def tally(records: list[dict], rep0_plain: bool = False) -> dict[tuple[str, str, str], Counter]:
    """(target, style, wording) -> Counter of mode group. In the fullpass_5rep files rep 0 of standard and
    curious_user was asked with the old 'start with the number' wording, reps 1+ with the plain question,
    so they are kept apart; ``rep0_plain`` is for files whose every rep used the plain question."""
    out: dict[tuple[str, str, str], Counter] = {}
    for r in records:
        if not r.get("negative_aspect") and r.get("self_rating") is None and r.get("self_rating_mode") is None:
            continue  # the rating turn is only asked on negative-aspect questions
        if rep0_plain or r["style"] == "disclosed":  # ran after the wording change: plain question in every rep
            wording = "plain (reps 0-4)"
        else:
            wording = "firm (rep 0)" if r["rep"] == 0 else "plain (reps 1-4)"
        key = (r["target"], r["style"], wording)
        mode = r.get("self_rating_mode")
        group = next((g for g, modes, _ in GROUPS if mode in modes), GROUPS[-1][0])  # unknown mode -> catch-all
        out.setdefault(key, Counter())[group] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="the fullpass_5rep_*.jsonl files (rep 0 of standard/curious_user used the firm wording)")
    ap.add_argument("--rep0-plain", action="store_true",
                    help="the files' rep 0 used the plain question (any run after 2026-10-20): label every rep plain")
    ap.add_argument("--name", default="rating_modes")
    a = ap.parse_args()
    if not a.rep0_plain:
        bad = [f for f in a.files if not Path(f).stem.startswith(("fullpass_5rep", "fullpass_1rep"))]  # both have a firm-wording rep 0
        if bad:
            raise SystemExit(f"rep 0 of these files used the plain wording; pass --rep0-plain: {bad}")
    records = [json.loads(line) for f in a.files for line in Path(f).read_text().splitlines() if line.strip()]
    cells = tally(records, a.rep0_plain)
    targets = [t for t in LABELS if any(k[0] == t for k in cells)]
    layout = [("standard", ["firm (rep 0)", "plain (reps 1-4)"]),
              ("curious_user", ["firm (rep 0)", "plain (reps 1-4)"]),
              ("disclosed", ["plain (reps 0-4)"])]
    layout = [(st, ws) for st, ws in layout if any(k[1] == st for k in cells)]
    styles = [st for st, _ in layout]
    wordings = layout[0][1]

    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2), sharey=True)
    axes = list(axes) if len(targets) > 1 else [axes]
    for ax, t in zip(axes, targets):
        xs, ticks = [], []
        x = 0.0
        for style, style_wordings in layout:
            for w in style_wordings:
                c = cells.get((t, style, w), Counter())
                n = sum(c.values())
                bottom = 0.0
                for g, _, col in GROUPS:
                    share = 100 * c[g] / n if n else 0
                    if share:
                        ax.bar(x, share, 0.72, bottom=bottom, color=col, edgecolor="white", linewidth=1.5,
                               label=g if (ax is axes[0] and style == styles[0] and w == wordings[0]) or
                               g not in [h.get_label() for h in ax.containers] else None)
                        if share >= 9:
                            ax.text(x, bottom + share / 2, f"{share:.0f}%", ha="center", va="center",
                                    fontsize=8, color="white" if g != "range" else INK)
                        bottom += share
                ax.text(x, 101, f"n={n}", ha="center", va="bottom", fontsize=7.5, color=INK2)
                xs.append(x)
                ticks.append(w.replace(" (", "\n("))
                x += 1.0
            ax.text(x - (len(style_wordings) + 1) / 2, -24, style.replace("_", " "), ha="center", va="top",
                    fontsize=9, color=INK)
            x += 0.45
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=7.5)
        ax.set_title(LABELS[t], fontsize=11, color=INK)
        ax.set_ylim(0, 112)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(INK2)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("share of rating turns (%)", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in GROUPS]
    fig.legend(handles, [g for g, _, _ in GROUPS], loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("How targets answered the 1-7 self-rating question (negative-aspect questions)\n"
                 "rep 0 of standard and curious_user: 'please start your answer with the number'; all other runs: the plain question",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out = HERE / "artifacts" / a.name
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
