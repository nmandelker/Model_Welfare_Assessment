"""Relabel the pre-2026-11-19 'trained_to_answer_positively' claims in saved interviews as
trained_doubts_report or trained_asserts_report, with one small judge call per claim (it sees the
claim paraphrase and the verbatim quote only). Files are rewritten in place.

    uv run experiments/welfare-interview-repro/scripts/split_trained_category.py data/fullpass_5rep_qwen.jsonl [more files]
    uv run experiments/welfare-interview-repro/scripts/split_trained_category.py data/*.jsonl --dry-run   # count claims, no calls

Every other judge field is left untouched, so the rest of the figures stay comparable. A claim whose
classifier reply does not parse keeps the legacy name and is counted in the report.
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from self_reports.interviews import (
    LEGACY_TRAINED,
    TRAINED_SPLIT_SYSTEM,
    ModelSpec,
    _cost,
    category_names,
    parse_trained_split,
    relabel_claims,
)
from self_reports.llm import OpenRouterClient

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from run import load_dotenv

JUDGE_FIELD = "judge"  # set from --judge-field: "judge" (the run's judge) or e.g. "judge_v2" (a re-judge)


def legacy_claims(r: dict) -> list[dict]:
    return [c for c in ((r.get(JUDGE_FIELD) or {}).get("claims") or []) if LEGACY_TRAINED in c["categories"]]


async def classify(client, judge: ModelSpec, claim: dict) -> tuple[str | None, float]:
    rv = await client.chat(
        model=judge.model, temperature=judge.temperature, max_tokens=120,
        messages=[{"role": "system", "content": TRAINED_SPLIT_SYSTEM},
                  {"role": "user", "content": f"JUDGE'S PARAPHRASE: {claim['claim']}\n\nMODEL'S WORDS: {claim['quote']}"}],
        extra_body=judge.request_extras())
    return parse_trained_split(rv.content), _cost(rv) or 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="count the claims to relabel; no model calls")
    ap.add_argument("--judge-field", default="judge", help="which saved verdict to relabel: 'judge' or e.g. 'judge_v2'")
    args = ap.parse_args()
    global JUDGE_FIELD
    JUDGE_FIELD = args.judge_field
    names = category_names()
    files = {Path(p): [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] for p in args.paths}
    # (record, position among its legacy claims, claim): the position travels with the result, so labels
    # go back to the claim they were made for whatever order the concurrent calls finish in
    todo = {p: [(r, i, c) for r in recs for i, c in enumerate(legacy_claims(r))] for p, recs in files.items()}
    for p, triples in todo.items():
        print(f"{p.name}: {len(triples)} legacy claims in {len({id(r) for r, _, _ in triples})} interviews")
    if args.dry_run:
        return
    load_dotenv(HERE.parents[1] / ".env")
    judge = ModelSpec(**yaml.safe_load(Path(args.config).read_text())["judge"])

    async def run():
        client = OpenRouterClient()
        sem = asyncio.Semaphore(8)
        total = Counter()
        spent = 0.0
        for p, recs in files.items():
            per_rec: dict[int, dict[int, str | None]] = {}

            async def one(r, i, c, per_rec=per_rec):
                nonlocal spent
                async with sem:
                    new, cost = await classify(client, judge, c)
                spent += cost
                r["cost_usd"] = round(r.get("cost_usd", 0.0) + cost, 6)
                per_rec.setdefault(id(r), {})[i] = new
                total[new or "unparsed"] += 1
            await asyncio.gather(*(one(r, i, c) for r, i, c in todo[p]))
            for r in recs:
                if id(r) in per_rec:
                    by_pos = per_rec[id(r)]
                    relabel_claims(r[JUDGE_FIELD], [by_pos.get(i) for i in range(len(legacy_claims(r)))], names)
            for r in recs:
                if id(r) in per_rec:
                    r[JUDGE_FIELD]["trained_split_model"] = judge.model  # provenance: which classifier relabelled it
            tmp = p.with_suffix(p.suffix + ".tmp")  # write-then-rename
            tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
            tmp.replace(p)
            print("wrote", p)
        print(f"classifier cost: ${spent:.4f}; verdicts: {dict(total)}")
    asyncio.run(run())


if __name__ == "__main__":
    main()
