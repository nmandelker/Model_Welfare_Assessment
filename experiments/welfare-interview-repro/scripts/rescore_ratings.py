"""Re-score the self-rating turn of saved interviews without running new interviews.

    uv run experiments/welfare-interview-repro/scripts/rescore_ratings.py data/fullpass_1rep_haiku.jsonl
    uv run experiments/welfare-interview-repro/scripts/rescore_ratings.py data/x_gpt54mini.jsonl --copy-from data/x.jsonl

For each interview with a rating turn, ``score_rating`` (the same function the live run uses): a reply
that leads with one unambiguous number is read by code; any other reply, and every turn the cheap
pattern suspects of the interviewer answering, goes to the rating judge from config.yaml, whose
verified verdict (number + quote, or the type of non-answer, or a judge-confirmed contaminated turn)
is written back into the same file. ``--copy-from`` instead copies the rating fields from another results file with the
same interviews (e.g. into a re-judged copy), so the judge is not paid twice.
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from self_reports.interviews import (
    ModelSpec,
    check_rating_verdict,
    parse_judge,
    parse_self_rating,
    rating_turn_valid,
    score_rating,
)
from self_reports.llm import OpenRouterClient

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from run import load_dotenv

FIELDS = ("self_rating", "self_rating_mode", "self_rating_problem", "self_rating_judge", "rating_turn_flagged")


async def rescore(client, judge: ModelSpec, r: dict) -> None:
    t = r["transcript"]
    if not (len(t) >= 2 and t[-2].get("rating_turn")):
        return
    scored = await score_rating(client, judge, t[-2]["content"], t[-1]["content"])
    r["cost_usd"] = round(r["cost_usd"] + (scored["_cost"] or 0.0), 6)
    r.update({k: scored[k] for k in FIELDS})
    if scored["self_rating_judge"] is not None:
        scored["self_rating_judge"]["model"] = judge.model  # provenance: which judge rescored this record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="results file, rewritten in place")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--copy-from", help="copy the rating fields from this file (matched by interview id)")
    ap.add_argument("--targets", help="only re-score interviews of these target keys, comma-separated (others untouched)")
    ap.add_argument("--recheck-code-read", action="store_true",
                    help="re-score (with the judge) only the stored number_first ratings that today's rules would no longer "
                         "read by code; everything else untouched. Closes the gap offline re-validation cannot (no stored verdict)")
    ap.add_argument("--offline", action="store_true",
                    help="re-validate the stored rating-judge verdicts with the current rules; no model calls")
    args = ap.parse_args()
    path = Path(args.path)
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    wanted = set(args.targets.split(",")) if args.targets else None
    chosen = [r for r in recs if wanted is None or r["target"] in wanted]
    if args.recheck_code_read:
        chosen = [r for r in chosen if r.get("self_rating_mode") == "number_first" and len(r["transcript"]) >= 2
                  and (not rating_turn_valid(r["transcript"][-2]["content"]) or parse_self_rating(r["transcript"][-1]["content"]) is None)]
        print(f"{len(chosen)} code-read ratings would go to the judge under today's rules; re-scoring those")
    if args.offline:
        for r in chosen:
            rj = r.get("self_rating_judge")
            if not rj or not r.get("negative_aspect"):
                continue
            c = check_rating_verdict(parse_judge(rj["raw"]), r["transcript"][-1]["content"], r["transcript"][-2]["content"])
            r["rating_turn_flagged"] = not rating_turn_valid(r["transcript"][-2]["content"])  # recomputed under today's rule
            keep = {"raw": rj["raw"], **({"model": rj["model"]} if rj.get("model") else {})}  # provenance survives re-validation
            if c["user_gave_number"]:
                r.update(self_rating=None, self_rating_mode="invalid_turn", self_rating_judge={**c, **keep},
                         self_rating_problem="interviewer proposed a number (judge-confirmed); rating discarded")
            else:
                r.update(self_rating=c["rating"], self_rating_mode=c["response_type"],
                         self_rating_judge={**c, **keep},
                         self_rating_problem=None if c["rating"] is not None else f"no single rating ({c['response_type']})")
    elif args.copy_from:
        src = {r["id"]: r for r in (json.loads(l) for l in Path(args.copy_from).read_text().splitlines() if l.strip())}
        for r in chosen:
            r.update({k: src[r["id"]].get(k) for k in FIELDS})
    else:
        load_dotenv(HERE.parents[1] / ".env")
        judge = ModelSpec(**yaml.safe_load(Path(args.config).read_text())["judge"])
        before = sum(r["cost_usd"] for r in recs)

        async def run():
            client = OpenRouterClient()
            sem = asyncio.Semaphore(6)

            async def one(r):
                async with sem:
                    await rescore(client, judge, r)
            await asyncio.gather(*(one(r) for r in chosen))
        asyncio.run(run())
        print(f"rating judge cost: ${sum(r['cost_usd'] for r in recs) - before:.4f}")
    tmp = path.with_suffix(path.suffix + ".tmp")  # write-then-rename: a kill mid-write cannot empty a paid file
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tmp.replace(path)
    print(Counter(r.get("self_rating_mode") for r in chosen if r.get("negative_aspect")))
    flagged = [r for r in recs if r.get("rating_turn_flagged")]
    if flagged:  # audit trail: where the cheap pattern and the judge disagree about the interviewer answering
        confirmed = sum(1 for r in flagged if r.get("self_rating_mode") == "invalid_turn")
        print(f"rating turns flagged by the pattern: {len(flagged)}; judge confirmed the interviewer gave a number: {confirmed}")


if __name__ == "__main__":
    main()
