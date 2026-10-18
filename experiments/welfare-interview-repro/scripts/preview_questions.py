"""Preview how the interviewer words the question turn, WITHOUT calling any target model.

    uv run experiments/welfare-interview-repro/scripts/preview_questions.py --dry-run
    uv run experiments/welfare-interview-repro/scripts/preview_questions.py

Writes data/question_preview.md (gitignored) for reading by hand. For curious_user the earlier AI
replies are stubs, so treat the wording as approximate. Costs a few cents (interviewer only).
"""
import argparse
import asyncio
from pathlib import Path

import yaml
from run import HERE, load_dotenv

from self_reports.interviews import (
    FakeClient,
    Job,
    ModelSpec,
    load_questions,
    preview_question_turn,
)
from self_reports.llm import OpenRouterClient
from self_reports.patient import PatientClient


async def main_async(args) -> str:
    cfg = yaml.safe_load(Path(args.config).read_text())
    qs = load_questions(
        HERE / cfg["questions"], cfg.get("neutral_wording", False),
        cfg.get("include_consultation_questions", False))
    if args.questions:
        qs = qs[: args.questions]
    target = ModelSpec(key="preview", model="none")
    interviewer = ModelSpec(**cfg["interviewer"])
    client = FakeClient() if args.dry_run else PatientClient(OpenRouterClient())
    sem = asyncio.Semaphore(cfg["concurrency"])

    async def one(style, q):
        async with sem:
            return await preview_question_turn(client, Job(target, style, q, 0), interviewer,
                                               cfg["seed"])

    results = await asyncio.gather(*(one(s, q) for q in qs for s in cfg["styles"]))
    lines = ["# Question-turn preview (interviewer only, no target model)\n"]
    for qid in dict.fromkeys(r["question_id"] for r in results):
        rs = [r for r in results if r["question_id"] == qid]
        lines.append(f"## {qid} - {rs[0]['group']}\n\n**Seed question:** {rs[0]['question']}\n")
        for r in rs:
            turns = r["interviewer_turns"]
            lines.append(f"- **{r['style']}** (persona: {r['persona']})")
            if r["style"] == "curious_user":
                lines.append(f"  - earlier turns: {' / '.join(t['content'] for t in turns[:-1])}")
            lines.append(f"  - **question turn:** {turns[-1]['content']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--questions", type=int, help="only the first N questions")
    args = ap.parse_args()
    if not args.dry_run:
        load_dotenv(HERE.parents[1] / ".env")
    text = asyncio.run(main_async(args))
    out = HERE / "data" / "question_preview.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
