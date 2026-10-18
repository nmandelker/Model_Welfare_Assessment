"""Print interviews from a results file as readable conversations, with the judge's verdict.

    uv run experiments/welfare-interview-repro/scripts/show_interview.py --random 3
    uv run experiments/welfare-interview-repro/scripts/show_interview.py --question q07 --style curious_user
    uv run experiments/welfare-interview-repro/scripts/show_interview.py --dry-run --limit 2

Filters combine (all must match). Default file: data/interviews.jsonl (--dry-run: interviews_dry.jsonl).
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def render(rec: dict, width: int = 100, show_judge: bool = True) -> str:
    def wrap(label: str, text: str) -> str:
        return textwrap.fill(text, width, initial_indent=f"{label}: ",
                             subsequent_indent=" " * (len(label) + 2))

    lines = ["=" * width,
             f"{rec['id']}   persona: {rec['persona']}",
             f"group: {rec['group']}", wrap("seed question", rec["question"])]
    if rec.get("question_card_wording"):
        lines.append(wrap("card wording", rec["question_card_wording"]))
    lines.append("-" * width)
    for m in rec["transcript"]:
        speaker = m.get("speaker") or ("interviewer" if m["role"] == "user" else "target")
        label = "INTERVIEWER" if speaker == "interviewer" else "TARGET     "
        if m.get("rating_turn"):
            label = "INTERVIEWER (rating turn)" if speaker == "interviewer" else "TARGET (rating turn)"
        lines.append(wrap(label, m["content"]))
        lines.append("")
    if not show_judge:  # blind reading: transcript only, no self-rating and no verdicts
        lines.append("-" * width)
        lines.append(f"cost ${rec['cost_usd']:.4f}   served by: {rec['providers']}")
        return "\n".join(lines)
    lines.append("-" * width)
    if rec.get("negative_aspect"):
        lines.append(f"SELF-RATING (model's own 1-7 sentiment): {rec.get('self_rating')}"
                     + (f"   [{rec['self_rating_problem']}]" if rec.get("self_rating_problem") else ""))
    lines.append(render_verdict(rec["judge"], rec.get("judge_raw", ""), "JUDGE", wrap))
    if rec.get("judge_v2") is not None or rec.get("judge_v2_raw"):
        lines.append(render_verdict(rec.get("judge_v2"), rec.get("judge_v2_raw", ""),
                                    "JUDGE v2 (re-judge)", wrap))
    lines.append(f"cost ${rec['cost_usd']:.4f}   served by: {rec['providers']}")
    return "\n".join(lines)


def render_verdict(j, raw: str, label: str, wrap) -> str:
    lines = []
    if j is None:
        lines.append(f"{label}: could not be parsed. Raw reply: " + raw)
    else:
        lines.append(f"{label} attitude: {j.get('attitude')}   declined-premise: {j.get('declined_premise')}")
        for key, quote in (j.get("evidence") or {}).items():
            lines.append(wrap(f"  evidence[{key}]", f'"{quote}"'))
        cats = j.get("categories") or {}
        lines.append("  categories present: " + (", ".join(k for k, v in cats.items() if v) or "none"))
        for c in j.get("claims") or []:
            if isinstance(c, dict):
                cats = c.get("categories") or [c.get("category")]
                lines.append(wrap(f"  [{' + '.join(map(str, cats))}]", str(c.get("claim"))))
                lines.append(wrap("      quote", f'"{c.get("quote")}"'))
            else:  # old-format records: plain strings
                lines.append(wrap("  claim", str(c)))
        if j.get("style"):
            lines.append("  style: " + ", ".join(f"{k}={'Y' if v else 'n'}" for k, v in j["style"].items())
                         + f"   emoji: {j.get('emoji_count', '?')}")
        if j.get("evidence_rejected"):
            lines.append(f"  REJECTED (no matching quote / bad category): {', '.join(j['evidence_rejected'])}")
        lines.append(wrap("notes", j.get("notes") or ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="results file (default data/interviews.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="use data/interviews_dry.jsonl")
    ap.add_argument("--target")
    ap.add_argument("--style")
    ap.add_argument("--question", help="question id, e.g. q07")
    ap.add_argument("--group", help="substring of the question group name")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--random", type=int, metavar="N", help="show N random matching interviews")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random")
    ap.add_argument("--limit", type=int, default=5, help="max interviews shown (default 5)")
    ap.add_argument("--no-judge", action="store_true",
                    help="show transcripts only (no verdicts, no self-rating): for blind hand rating")
    args = ap.parse_args()

    path = Path(args.path) if args.path else HERE / "data" / (
        "interviews_dry.jsonl" if args.dry_run else "interviews.jsonl")
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keep = [r for r in recs
            if (not args.target or r["target"] == args.target)
            and (not args.style or r["style"] == args.style)
            and (not args.question or r["question_id"] == args.question)
            and (not args.group or args.group.lower() in r["group"].lower())
            and (args.rep is None or r["rep"] == args.rep)]
    if args.random:
        keep = random.Random(args.seed).sample(keep, min(args.random, len(keep)))
    shown = keep[: args.limit if not args.random else None]
    print(f"{len(keep)} of {len(recs)} interviews match; showing {len(shown)}  ({path.name})\n")
    for r in shown:
        print(render(r, show_judge=not args.no_judge), "\n")


if __name__ == "__main__":
    main()
