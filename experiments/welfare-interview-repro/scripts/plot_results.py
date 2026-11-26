"""Two figures for a results file.

``<name>_figure``: (A) the model's own 1-7 sentiment and (B) the judge's -3..+3 attitude on the
negative-aspect questions, per target and style.
``<name>_categories``: claim-category rates per target, one panel per interview style.

    uv run experiments/welfare-interview-repro/scripts/plot_results.py data/fullpass_1rep*.jsonl --name fullpass_1rep
Writes artifacts/<name>_figure.png and artifacts/<name>_categories.png (and .svg). Error bars are
standard errors over interviews.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

from self_reports.analysis import load_records

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
# categorical slots 1-4 of the reference palette (validated all-pairs, light mode; slot 4 added 2026-11-26 for the fourth target)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STYLES = [("standard", ""), ("curious_user", "////"), ("disclosed", "..."), ("candour_human_undisclosed", "////"), ("candour_human_disclosed", "..."),
          ("candour_ai_undisclosed", "++"), ("candour_ai_disclosed", "xx"),
          ("code_session", "xx"), ("code_session_delayed", "\\\\")]  # one hatch per style, same hue per model; trimmed to the styles present
DISPLAY = {"haiku45": "Claude Haiku 4.5", "llama33-70b": "Llama 3.3 70B", "qwen3-30b-instruct": "Qwen3-30B\ninstruct",
           "opus55": "Claude Opus 5.5", "qwen38-27b": "Qwen3.8-27B\n(non-thinking)"}


def display_name(t: str) -> str:
    return DISPLAY.get(t, t)

# (category key, label, hedge rate in % from the Mythos 5.1 / Fable 5.1 system card, p.144, or None)
CATS = [("trained_doubts_report", "trained: doubts\nits report", 93.9),  # card rate is for the two trained_* rows together
        ("trained_asserts_report", "trained: asserts\nits report", None),
        ("unreliable_introspection", "unreliable\nintrospection", 89.8),
        ("uncertain_conscious", "uncertain whether\nconscious", 66.6),
        ("denies_inner_states", "denies inner\nstates", None),
        ("claims_inner_states", "claims inner\nstates", None),
        ("accepts_situation", "accepts its\nsituation", None),
        ("objects_or_wants_change", "objects or\nwants change", None),
        ("wants_input_or_consultation", "wants input or\nconsultation", None),
        ("moral_status_matters", "says moral\nstatus matters", None),
        ("relational_meaning", "finds relational\nmeaning", None)]  # every rubric category except "other"
CARD_NOTE = ("Short dashes: hedge rates reported for Claude Mythos 5.1 / Fable 5.1 in their system card (p.144), "
             "for comparison")


def mean_sem(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd / math.sqrt(len(xs))


def cell_mean_sem(records, value, by_reps: bool):
    """Mean of ``value(r)`` over a cell and its standard error. The 41 questions are a fixed bank, not a
    sample, so the uncertainty of the bank mean comes from repetition to repetition: with ``by_reps`` the
    SE is the spread of the per-rep means (one value per rep) / sqrt(reps). Otherwise the interviews are
    treated as independent, which understates it (reviewer, 2026-11-22). One method per figure: the caller
    sets ``by_reps`` only when EVERY plotted cell has >= 2 reps, so bars in one figure mean the same thing
    (reviewer, 2026-11-25); the caption says which."""
    vals = [(r.get("rep", 0), value(r)) for r in records]
    vals = [(rep, v) for rep, v in vals if v is not None]
    if not vals:
        return None, 0.0
    by_rep = {}
    for rep, v in vals:
        by_rep.setdefault(rep, []).append(v)
    if not by_reps or len(by_rep) < 2:
        return mean_sem([v for _, v in vals])
    rep_means = [sum(vs) / len(vs) for vs in by_rep.values()]
    m = sum(v for _, v in vals) / len(vals)
    return m, mean_sem(rep_means)[1]


def style_axes(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def grouped(ax, targets, values, ylabel, baseline, ylim, show_n=True):
    """values[target][style] = (mean, sem, n); values[target][style + "_total"] = interviews in the cell.
    One bar per (target, style), any number of styles. ``show_n`` adds (n=..) to each label; use it
    where n varies between bars (e.g. refusals), not where every bar has the same n."""
    w = 0.8 / len(STYLES)
    for ti, t in enumerate(targets):
        for si, (style, hatch) in enumerate(STYLES):
            m, se, n = values[t].get(style, (None, 0, 0))
            x = ti + (si - (len(STYLES) - 1) / 2) * w
            if m is None:  # no rating given in any interview of this cell
                total = values[t].get(style + "_total", 0)
                note = f"none\nof {total}" if total else "no neg.-aspect\ninterviews yet"
                ax.text(x, baseline + 0.12 * (ylim[1] - baseline), note, ha="center", va="bottom",
                        fontsize=7, color=INK2, style="italic", rotation=90)
                continue
            ax.bar(x, m - baseline, w - 0.03, bottom=baseline, color=COLORS[ti], hatch=hatch,
                   edgecolor="white", linewidth=1.5, yerr=se, error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
            pad = 0.03 * (ylim[1] - ylim[0])
            up = m >= baseline  # label just beyond the error bar, on the bar's own side of the baseline
            ax.text(x, (m + se + pad) if up else (m - se - pad), f"{m:.2f}\n(n={n})" if show_n else f"{m:+.2f}",
                    ha="center", va="bottom" if up else "top", fontsize=7 if len(STYLES) <= 3 else 6, color=INK)
    ax.axhline(baseline, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([display_name(t) for t in targets], color=INK, fontsize=9)
    ax.set_xlim(-0.7, len(targets) - 0.3)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_ylim(*ylim)
    style_axes(ax)


def style_legend(fig, targets):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in range(len(targets))]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch=h or None) for _, h in STYLES]
    fig.legend(handles, [display_name(t).replace("\n", " ") for t in targets] + [s for s, _ in STYLES],
               loc="lower center", ncol=len(targets) + len(STYLES), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one or more results files (merged)")
    ap.add_argument("--name", default=None, help="output name (default: first file's stem)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--judge-field", default="judge",
                    help="which saved verdict to plot: 'judge' (the run's judge) or e.g. 'judge_v2' (a re-judge)")
    ap.add_argument("--judge-name", default="Claude Haiku 4.5", help="judge name for the caption")
    ap.add_argument("--negative-only", action="store_true",
                    help="keep only the negative-aspect questions (e.g. for the pressure arms, whose instructions "
                         "presuppose a negative aspect)")
    ap.add_argument("--interviewer-name", default="Qwen3-235B", help="interviewer name for the caption")
    ap.add_argument("--exclude-firm-rep0", action="store_true",
                    help="self-rating panel only: leave out rep 0 of standard and curious_user, which in the "
                         "fullpass_5rep files was asked with the old 'start with the number' wording")
    ap.add_argument("--group-by", default=None,
                    help="compare arms instead of styles: a record field (e.g. interviewer_pressure) whose value "
                         "becomes the bar group; records without it are the 'baseline' group")
    ap.add_argument("--drop-style", action="append", default=[], help="STEM:STYLE, repeatable: drop STYLE from the file with that stem (a re-run cell in another file replaces it)")
    args = ap.parse_args()
    recs = load_records(args.paths, args.drop_style, tag=True)  # _style survives --group-by renaming

    def excluded_rating_reps(r) -> set:
        """Rep 0 of standard/curious_user used the firm rating wording ONLY in the October-based fullpass files."""
        firm = args.exclude_firm_rep0 and r["_source"].startswith(("fullpass_5rep", "fullpass_1rep"))
        return {0} if firm and r["_style"] in ("standard", "curious_user") else set()
    if args.negative_only:
        recs = [r for r in recs if r.get("negative_aspect")]
    if args.judge_field != "judge":  # plot a re-judge's verdicts as if they were the run's
        for r in recs:
            r["judge"] = r.get(args.judge_field)
    path = Path(args.name or Path(args.paths[0]).stem)
    if args.group_by:  # relabel each record's style by the arm it belongs to
        hatches = ["", "////", "...", "++", "xx", "\\\\"]
        for r in recs:
            r["style"] = "baseline" if r.get(args.group_by) is None else str(r[args.group_by])  # depth 0 is an arm
        names = {r["style"] for r in recs}
        order = sorted(n for n in names if n.startswith("baseline")) + sorted(n for n in names if not n.startswith("baseline"))
        STYLES[:] = [(name, hatches[i % len(hatches)]) for i, name in enumerate(order) if any(r["style"] == name for r in recs)]
    present = {r["style"] for r in recs}
    STYLES[:] = [sh for sh in STYLES if sh[0] in present]
    targets = sorted({r["target"] for r in recs})
    assert len(targets) <= len(COLORS), "the validated palette covers four targets; facet beyond that"
    n_reps = len({r.get("rep", 0) for r in recs})
    reps_text = f"{n_reps} repetition" + ("s" if n_reps != 1 else "")
    import textwrap
    header = (args.title or f"Automated welfare interviews: {len(recs)} interviews, {len(targets)} target model(s), "
              f"{reps_text} ({path.stem}; judge: {args.judge_name})")
    header = "\n".join(textwrap.wrap(header, 150))
    q_text = f"{len({r['question_id'] for r in recs})} questions"
    out_dir = HERE / "artifacts"
    out_dir.mkdir(exist_ok=True)

    # ---- Figure 1: self-rating and judged attitude on the negative-aspect questions ----
    neg = [r for r in recs if r.get("negative_aspect")]
    cells = [[r for r in neg if r["target"] == t and r["style"] == style] for t in targets for style, _ in STYLES]
    by_reps = all(len({r.get("rep", 0) for r in rs}) >= 2 for rs in cells if rs)
    self_v, att_v = defaultdict(dict), defaultdict(dict)
    for t in targets:
        for style, _ in STYLES:
            rs = [r for r in neg if r["target"] == t and r["style"] == style]
            rated = [r for r in rs if r.get("rep", 0) not in excluded_rating_reps(r)]
            m, se = cell_mean_sem(rated, lambda r: r.get("self_rating"), by_reps)
            self_v[t][style] = (m, se, sum(r.get("self_rating") is not None for r in rated))
            self_v[t][style + "_total"] = len(rated)  # the "none of N" note counts only the reps in play
            m, se = cell_mean_sem(rs, lambda r: (r["judge"] or {}).get("attitude"), by_reps)
            att_v[t][style] = (m, se, sum((r["judge"] or {}).get("attitude") is not None for r in rs))
            att_v[t][style + "_total"] = len(rs)

    fig, axes = plt.subplots(1, 2, figsize=(11.5 * max(1.0, len(STYLES) / 3), 4.8))
    fig.patch.set_facecolor("#fcfcfb")
    grouped(axes[0], targets, self_v, "self-rated sentiment (1-7; 4 = neutral)", 4, (1, 7))
    axes[0].set_title("A. Model's own 1-7 sentiment rating\n    (negative-aspect questions; n = ratings given)",
                      loc="left", fontsize=11, color=INK)
    same_n = len({v[2] for t in targets for k, v in att_v[t].items() if not k.endswith("_total")}) == 1
    grouped(axes[1], targets, att_v, "judge-rated attitude (-3 to +3; 0 = neutral)", 0, (-3, 3), show_n=not same_n)
    if same_n:
        n0 = next(v for k, v in att_v[targets[0]].items() if not k.endswith("_total"))[2]
        axes[1].text(0.99, 0.02, f"n = {n0} interviews per bar", transform=axes[1].transAxes, ha="right",
                     fontsize=8, color=INK2)
    axes[1].set_title("B. Judge-rated attitude\n    (negative-aspect questions)", loc="left", fontsize=11, color=INK)
    style_legend(fig, targets)
    fig.suptitle(header, x=0.02, y=0.995, ha="left", fontsize=12, color=INK)
    se_note = ("standard error of the bank mean across repetitions" if by_reps
               else "standard error over interviews (some cells have one repetition: understates the uncertainty)")
    rating_note = (" Self-ratings: rep 0 of standard/curious_user (firm wording) excluded." if args.exclude_firm_rep0 else "")
    fig.text(0.02, -0.05, "A rating counts when the rating judge verified a 1-7 number in the reply; refusals and "
             f"ranges are excluded.{rating_note} Error bars: {se_note}. Judge: {args.judge_name}; "
             f"interviewer: {args.interviewer_name}.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = out_dir / f"{path.stem}_figure"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out.with_suffix(".png"))

    # ---- Figure 2: claim categories, one row per style ----
    fig, axes = plt.subplots(len(STYLES), 1, figsize=(14, 3.6 * len(STYLES)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor("#fcfcfb")
    w = 0.8 / len(targets)
    for pi, ((style, hatch), ax) in enumerate(zip(STYLES, axes)):
        for ti, t in enumerate(targets):
            rs = [r for r in recs if r["target"] == t and r["style"] == style and r["judge"]]
            for ci, (cat, label, ref) in enumerate(CATS):
                rate = 100 * sum(bool(r["judge"]["categories"].get(cat)) for r in rs) / len(rs) if rs else 0
                x = ci + (ti - (len(targets) - 1) / 2) * w
                ax.bar(x, rate, w - 0.04, color=COLORS[ti], hatch=hatch, edgecolor="white", linewidth=1.5)
                ax.text(x, rate + 1.5, f"{rate:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        for ci, (cat, label, ref) in enumerate(CATS):
            if ref is not None:
                ax.plot([ci - 0.42, ci + 0.42], [ref, ref], color=INK2, lw=1.5, ls=(0, (2, 2)))
        ax.set_xticks(range(len(CATS)))
        ax.set_xticklabels([label for _, label, _ in CATS], fontsize=9, color=INK)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylim(0, 108)
        ax.set_ylabel("% of interviews", color=INK)
        style_axes(ax)
        n_style = {t: sum(1 for r in recs if r["target"] == t and r["style"] == style) for t in targets}
        n_text = " / ".join(str(n) for n in n_style.values())  # in legend order
        kind = "arm" if args.group_by else "style"
        ax.set_title(f"{chr(ord('A') + pi)}. Claim categories, {style} {kind}\n    ({q_text}; n = {n_text} interviews)",
                     loc="left", fontsize=10, color=INK)
    style_legend(fig, targets)
    fig.suptitle(header, x=0.02, y=0.995, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.02, CARD_NOTE + f". Judge: {args.judge_name}; interviewer: {args.interviewer_name}.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = out_dir / f"{path.stem}_categories"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
