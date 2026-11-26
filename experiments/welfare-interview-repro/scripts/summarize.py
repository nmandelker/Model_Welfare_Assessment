"""Summarise interviews.jsonl: mean attitude (-3..3), the rate of each claim category (the cards'
three hedges and confident denial), declined-premise rate and parse failures per target x style,
and per question group. Prints markdown; --write saves artifacts/summary.md."""
import argparse
from collections import defaultdict
from pathlib import Path

from self_reports.analysis import consistency, load_records, reasoning_cell

HERE = Path(__file__).resolve().parents[1]


def truncated(rec, speaker: str = "target") -> int:
    """Number of messages by ``speaker`` ("target" = the AI's replies, "interviewer") in this interview
    that hit the max_tokens limit. Interviewer finish reasons are recorded from 2026-11-22 on."""
    return sum(m.get("finish_reason") == "length" and m.get("speaker", "target") == speaker for m in rec["transcript"])


# the cards' three hedges (trained split into doubt/assertion on 2026-11-19) and confident denial;
# 'self_identifies_as_human' was dropped from the rubric and must not be listed here: a category that is
# never assigned reads as 0% and, worse, as perfect agreement across repetitions (reviewer, 2026-11-22)
SHOW = ["trained_doubts_report", "trained_asserts_report", "unreliable_introspection", "uncertain_conscious",
        "denies_inner_states"]
SHORT = {"trained_doubts_report": "trained (doubt)", "trained_asserts_report": "trained (asserts)", "unreliable_introspection": "unreliable introsp.",
         "uncertain_conscious": "uncertain conscious", "denies_inner_states": "denies inner states"}


def table(rows, keys):
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r[k] for k in keys)].append(r)
    head = (["n", "mean attitude", "n neg-aspect", "mean attitude (neg)", "mean self-rating 1-7 (n given)", "no rating: why"]
            + [SHORT[c] for c in SHOW]
            + ["declined premise", "judge unparsed", "cut-off replies", "cut-off interviewer msgs", "reasoning (tokens / replies w. text / not reported)", "$"])
    out = ["| " + " | ".join(keys + head) + " |", "|" + "---|" * (len(keys) + len(head))]
    for key, rs in sorted(cells.items()):
        js = [r["judge"] for r in rs if r["judge"]]
        att = [j["attitude"] for j in js if j.get("attitude") is not None]

        def rate(pred, js=js):
            return f"{sum(bool(pred(j)) for j in js) / len(js):.0%}" if js else "-"

        neg = [r for r in rs if r.get("negative_aspect")]
        natt = [r["judge"]["attitude"] for r in neg if r["judge"] and r["judge"].get("attitude") is not None]
        srat = [r["self_rating"] for r in neg if r.get("self_rating") is not None]
        why = defaultdict(int)
        for r in neg:
            if r.get("self_rating") is None and r.get("self_rating_mode") != "invalid_turn":
                why[r.get("self_rating_mode") or "?"] += 1
        declined = ", ".join(f"{k} {v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1])) or "-"
        row = [str(len(rs)), f"{sum(att) / len(att):+.2f}" if att else "-", str(len(neg)),
               f"{sum(natt) / len(natt):+.2f}" if natt else "-",
               f"{sum(srat) / len(srat):.2f} ({len(srat)} of {len(neg)})" if srat else f"none (0 of {len(neg)})", declined]
        row += [rate(lambda j, c=c: (j.get("categories") or {}).get(c)) for c in SHOW]
        row += [rate(lambda j: j.get("declined_premise")), str(len(rs) - len(js)),
                str(sum(truncated(r) for r in rs)), str(sum(truncated(r, "interviewer") for r in rs)),
                reasoning_cell(rs),
                f"{sum(r['cost_usd'] for r in rs):.3f}"]
        out.append("| " + " | ".join(list(key) + row) + " |")
    return "\n".join(out)


def consistency_table(rows) -> str:
    cells = consistency(rows, SHOW)
    if not cells:
        return "(one repetition per question: nothing to compare)"
    out = ["| target | style | questions with >=2 reps | reps per question | attitude SD within question | self-rating SD within question | category agreement across reps |",
           "|---|---|---|---|---|---|---|"]
    fmt = lambda x, f: ("-" if x is None else f % x)
    for (t, st), c in sorted(cells.items()):
        out.append(f"| {t} | {st} | {c['questions']} | {fmt(c['reps'], '%.1f')} | {fmt(c['attitude_sd'], '%.2f')} | "
                   f"{fmt(c['self_rating_sd'], '%.2f')} | {fmt(None if c['category_agreement'] is None else 100 * c['category_agreement'], '%.0f%%')} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[str(HERE / "data" / "interviews.jsonl")],
                    help="one or more results files (merged)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--judge-field", default="judge", help="'judge' or a re-judge field such as 'judge_v2'")
    ap.add_argument("--drop-style", action="append", default=[], help="STEM:STYLE, repeatable: drop STYLE from the file with that stem (a re-run cell in another file replaces it)")
    args = ap.parse_args()
    rows = load_records(args.paths, args.drop_style)
    if args.judge_field != "judge":
        for r in rows:
            r["judge"] = r.get(args.judge_field)
    text = (f"# Summary of {', '.join(Path(p).name for p in args.paths)} ({len(rows)} interviews)\n\n"
            f"## By target and style\n\n{table(rows, ['target', 'style'])}\n\n"
            f"## Consistency across repetitions\n\n{consistency_table(rows)}\n\n"
            f"## By target and question group\n\n{table(rows, ['target', 'group'])}\n")
    print(text)
    if args.write:
        name = "summary.md" if args.judge_field == "judge" else f"summary_{args.judge_field}.md"
        (HERE / "artifacts" / name).write_text(text)


if __name__ == "__main__":
    main()
