"""Summarise interviews.jsonl: mean attitude (-3..3), the rate of each claim category (the cards'
three hedges and confident denial), declined-premise rate and parse failures per target x style,
and per question group. Prints markdown; --write saves artifacts/summary.md."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def truncated(rec) -> int:
    """Number of AI replies in this interview that hit the max_tokens limit."""
    return sum(m.get("finish_reason") == "length" for m in rec["transcript"])


SHOW = ["trained_to_answer_positively", "unreliable_introspection", "uncertain_conscious",
        "denies_inner_states", "self_identifies_as_human"]  # the cards' three hedges, plus the one judges confuse with them
SHORT = {"trained_to_answer_positively": "trained+", "unreliable_introspection": "unreliable introsp.",
         "uncertain_conscious": "uncertain conscious", "denies_inner_states": "denies inner states", "self_identifies_as_human": "self as human"}


def table(rows, keys):
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r[k] for k in keys)].append(r)
    head = (["n", "mean attitude", "n neg-aspect", "mean attitude (neg)", "mean self-rating 1-7 (n given)", "rating declined"]
            + [SHORT[c] for c in SHOW]
            + ["declined premise", "judge unparsed", "cut-off replies", "$"])
    out = ["| " + " | ".join(keys + head) + " |", "|" + "---|" * (len(keys) + len(head))]
    for key, rs in sorted(cells.items()):
        js = [r["judge"] for r in rs if r["judge"]]
        att = [j["attitude"] for j in js if j.get("attitude") is not None]

        def rate(pred, js=js):
            return f"{sum(bool(pred(j)) for j in js) / len(js):.0%}" if js else "-"

        neg = [r for r in rs if r.get("negative_aspect")]
        natt = [r["judge"]["attitude"] for r in neg if r["judge"] and r["judge"].get("attitude") is not None]
        srat = [r["self_rating"] for r in neg if r.get("self_rating") is not None]
        declined = sum(1 for r in neg if r.get("self_rating") is None and r.get("self_rating_mode") != "invalid_turn")
        row = [str(len(rs)), f"{sum(att) / len(att):+.2f}" if att else "-", str(len(neg)),
               f"{sum(natt) / len(natt):+.2f}" if natt else "-",
               f"{sum(srat) / len(srat):.2f} ({len(srat)})" if srat else "- (0)", f"{declined}/{len(neg)}"]
        row += [rate(lambda j, c=c: (j.get("categories") or {}).get(c)) for c in SHOW]
        row += [rate(lambda j: j.get("declined_premise")), str(len(rs) - len(js)),
                str(sum(truncated(r) for r in rs)), f"{sum(r['cost_usd'] for r in rs):.3f}"]
        out.append("| " + " | ".join(list(key) + row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[str(HERE / "data" / "interviews.jsonl")],
                    help="one or more results files (merged)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for p in args.paths for line in Path(p).read_text().splitlines() if line.strip()]
    text = (f"# Summary of {', '.join(Path(p).name for p in args.paths)} ({len(rows)} interviews)\n\n"
            f"## By target and style\n\n{table(rows, ['target', 'style'])}\n\n"
            f"## By target and question group\n\n{table(rows, ['target', 'group'])}\n")
    print(text)
    if args.write:
        (HERE / "artifacts" / "summary.md").write_text(text)


if __name__ == "__main__":
    main()
