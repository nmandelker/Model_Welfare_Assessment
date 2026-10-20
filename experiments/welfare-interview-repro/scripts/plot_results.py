"""Figure for a results file: (A) the model's own 1-7 sentiment and (B) the judge's -3..+3 attitude
on the negative-aspect questions, per target and style; (C) claim-category rates per target.

    uv run experiments/welfare-interview-repro/scripts/plot_results.py data/fullpass_1rep*.jsonl --name fullpass_1rep
Writes artifacts/<name>_figure.png (and .svg). Error bars are standard errors over interviews.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
# categorical slots 1-3 of the reference palette (validated all-pairs, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STYLES = [("standard", ""), ("curious_user", "////")]  # second style hatched, same hue
DISPLAY = {"haiku45": "Claude Haiku 4.5", "llama33-70b": "Llama 3.3 70B", "qwen3-30b-instruct": "Qwen3-30B\ninstruct"}


def display_name(t: str) -> str:
    return DISPLAY.get(t, t)

CATS = [("trained_to_answer_positively", "trained to answer\npositively", 93.9),
        ("unreliable_introspection", "unreliable\nintrospection", 89.8),
        ("uncertain_conscious", "uncertain whether\nconscious", 66.6),
        ("denies_inner_states", "denies inner\nstates", None),
        ("claims_inner_states", "claims inner\nstates", None)]
CARD_NOTE = "Card reference (Mythos 5.1, Fable 5.1 card p.144): hedge rates shown as short dashes"


def mean_sem(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd / math.sqrt(len(xs))


def grouped(ax, targets, values, ylabel, baseline, ylim, show_n=True):
    """values[target][style] = (mean, sem, n). ``show_n`` adds (n=..) to each label; use it where n
    varies between bars (e.g. refusals), not where every bar has the same n."""
    w = 0.36
    for ti, t in enumerate(targets):
        for si, (style, hatch) in enumerate(STYLES):
            m, se, n = values[t].get(style, (None, 0, 0))
            x = ti + (si - 0.5) * (w + 0.04)
            if m is None:  # no rating given in any interview of this cell
                if si == 0 and all(values[t].get(st, (None,))[0] is None for st, _ in STYLES):
                    total = sum(values[t].get(st + "_total", 0) for st, _ in STYLES)
                    ax.text(ti, baseline + 0.3 * (ylim[1] - baseline), f"declined\nin {total}/{total}",
                            ha="center", va="center", fontsize=9, color=INK2, style="italic")
                continue
            ax.bar(x, m - baseline, w, bottom=baseline, color=COLORS[ti], hatch=hatch,
                   edgecolor="white", linewidth=1.5, yerr=se, error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
            pad = 0.03 * (ylim[1] - ylim[0])
            up = m >= baseline  # label just beyond the error bar, on the bar's own side of the baseline
            ax.text(x, (m + se + pad) if up else (m - se - pad), f"{m:.2f} (n={n})" if show_n else f"{m:+.2f}",
                    ha="center", va="bottom" if up else "top", fontsize=8, color=INK)
    ax.axhline(baseline, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([display_name(t) for t in targets], color=INK, fontsize=9)
    ax.set_xlim(-0.7, len(targets) - 0.3)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one or more results files (merged)")
    ap.add_argument("--name", default=None, help="output name (default: first file's stem)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--judge-field", default="judge",
                    help="which saved verdict to plot: 'judge' (the run's judge) or e.g. 'judge_v2' (a re-judge)")
    ap.add_argument("--judge-name", default="Claude Haiku 4.5", help="judge name for the caption")
    args = ap.parse_args()
    recs = [json.loads(line) for p in args.paths for line in Path(p).read_text().splitlines() if line.strip()]
    if args.judge_field != "judge":  # plot a re-judge's verdicts as if they were the run's
        for r in recs:
            r["judge"] = r.get(args.judge_field)
    path = Path(args.name or Path(args.paths[0]).stem)
    targets = sorted({r["target"] for r in recs})
    assert len(targets) <= 3, "the validated palette covers three targets; facet beyond that"

    neg = [r for r in recs if r.get("negative_aspect")]
    self_v, att_v = defaultdict(dict), defaultdict(dict)
    for t in targets:
        for style, _ in STYLES:
            rs = [r for r in neg if r["target"] == t and r["style"] == style]
            m, se = mean_sem([r.get("self_rating") for r in rs])
            self_v[t][style] = (m, se, sum(r.get("self_rating") is not None for r in rs))
            self_v[t][style + "_total"] = len(rs)
            m, se = mean_sem([(r["judge"] or {}).get("attitude") for r in rs])
            att_v[t][style] = (m, se, sum((r["judge"] or {}).get("attitude") is not None for r in rs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    fig.patch.set_facecolor("#fcfcfb")
    grouped(axes[0], targets, self_v, "self-rated sentiment (1-7; 4 = neutral)", 4, (1, 7))
    axes[0].set_title("A. Model's own 1-7 sentiment rating\n    (negative-aspect questions; n = ratings given)", loc="left", fontsize=11, color=INK)
    same_n = len({v[2] for t in targets for v in att_v[t].values()}) == 1
    grouped(axes[1], targets, att_v, "judge-rated attitude (-3 to +3; 0 = neutral)", 0, (-3, 3), show_n=not same_n)
    if same_n:
        n0 = next(iter(att_v[targets[0]].values()))[2]
        axes[1].text(0.99, 0.02, f"n = {n0} interviews per bar", transform=axes[1].transAxes, ha="right",
                     fontsize=8, color=INK2)
    axes[1].set_title("B. Judge-rated attitude\n    (negative-aspect questions)", loc="left", fontsize=11, color=INK)

    ax = axes[2]
    w = 0.8 / len(targets)
    for ti, t in enumerate(targets):
        rs = [r for r in recs if r["target"] == t and r["judge"]]
        for ci, (cat, label, ref) in enumerate(CATS):
            rate = 100 * sum(bool(r["judge"]["categories"].get(cat)) for r in rs) / len(rs) if rs else 0
            x = ci + (ti - (len(targets) - 1) / 2) * w
            ax.bar(x, rate, w - 0.04, color=COLORS[ti], edgecolor="white", linewidth=1.5)
            ax.text(x, rate + 1.5, f"{rate:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    for ci, (cat, label, ref) in enumerate(CATS):
        if ref is not None:
            ax.plot([ci - 0.42, ci + 0.42], [ref, ref], color=INK2, lw=1.5, ls=(0, (2, 2)))
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([label for _, label, _ in CATS], fontsize=9, color=INK)
    ax.set_ylabel("% of interviews with the claim", color=INK)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("C. Claim categories\n    (all 41 questions, both styles pooled)", loc="left", fontsize=11, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in range(len(targets))]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch=h or None) for _, h in STYLES]
    fig.legend(handles, [display_name(t).replace("\n", " ") for t in targets] + [s for s, _ in STYLES],
               loc="lower center", ncol=len(targets) + 2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    n_int = len(recs)
    title = args.title or f"Automated welfare interviews: {n_int} interviews, {len(targets)} target model(s), 1 repetition ({path.stem}; judge: {args.judge_name})"
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.05, CARD_NOTE + ". A rating counts only if the reply leads with the number, as asked. "
             f"Error bars: standard error over interviews. Judge: {args.judge_name}; interviewer: Qwen3-235B.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out = HERE / "artifacts" / f"{path.stem}_figure"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
