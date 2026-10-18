"""Run the welfare-interview experiment from config.yaml.

    uv run experiments/welfare-interview-repro/scripts/run.py --dry-run
    uv run experiments/welfare-interview-repro/scripts/run.py --targets qwen3-32b --questions 2 --reps 1
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from self_reports.interviews import FakeClient, Job, ModelSpec, load_questions, run_all
from self_reports.llm import OpenRouterClient
from self_reports.patient import PatientClient

HERE = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=value lines); does not override variables already set."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def git_state() -> dict:
    """Commit the code is at, and whether there are uncommitted changes (then the commit alone
    does not pin down the code that ran)."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None, "branch": git("branch", "--show-current"),
            "uncommitted_changes": bool(git("status", "--porcelain"))}


def key_usage() -> float:
    """Total dollars OpenRouter says this API key has spent (an independent check on our own
    per-call cost sum). May lag by a few seconds."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return float(json.load(resp)["data"]["usage"])


def log_run(path: Path, record: dict) -> None:
    """Append one line to the run log kept next to the results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"time": datetime.now(UTC).isoformat(), **record}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="fake models, no network, no cost")
    ap.add_argument("--targets", nargs="*", help="only these target keys")
    ap.add_argument("--questions", type=int, help="only the first N questions")
    ap.add_argument("--question-ids", help="only these ids, comma-separated, e.g. q08,q13,q25")
    ap.add_argument("--reps", type=int, help="override reps")
    ap.add_argument("--max-cost", type=float, help="override max_cost_usd")
    ap.add_argument("--out", help="output JSONL (default data/interviews[_dry].jsonl)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    questions = load_questions(
        HERE / cfg["questions"], cfg.get("neutral_wording", False),
        cfg.get("include_consultation_questions", False))
    if args.questions:
        questions = questions[: args.questions]
    if args.question_ids:
        wanted = set(args.question_ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    targets = [ModelSpec(**t) for t in cfg["targets"]
               if not args.targets or t["key"] in args.targets]
    jobs = [Job(t, style, q, rep) for t in targets for style in cfg["styles"]
            for q in questions for rep in range(args.reps or cfg["reps"])]

    if args.dry_run:
        client = FakeClient()
    else:
        load_dotenv(HERE.parents[1] / ".env")
        retry = cfg.get("retry", {})
        client = PatientClient(OpenRouterClient(), interval_s=retry.get("interval_s", 60),
                               max_wait_s=retry.get("max_wait_min", 30) * 60)
    out = Path(args.out) if args.out else HERE / "data" / (
        "interviews_dry.jsonl" if args.dry_run else "interviews.jsonl")
    max_cost = args.max_cost if args.max_cost is not None else cfg["max_cost_usd"]
    run_log = out.with_name(out.stem + "_runs.jsonl")
    usage_before = None if args.dry_run else key_usage()
    log_run(run_log, {
        "event": "start", "key_usage_before_usd": usage_before, "command": sys.argv, "dry_run": args.dry_run,
        "config_file": str(Path(args.config).resolve().relative_to(HERE.parents[1])),
        "config": cfg, "effective": {"targets": [t.key for t in targets],
                                     "n_questions": len(questions), "n_jobs": len(jobs),
                                     "reps": args.reps or cfg["reps"], "max_cost_usd": max_cost},
        "git": git_state()})
    state = asyncio.run(run_all(
        client, jobs, ModelSpec(**cfg["interviewer"]), ModelSpec(**cfg["judge"]), out,
        concurrency=cfg["concurrency"], seed=cfg["seed"], max_cost_usd=max_cost,
        target_system_prompt=cfg["target_system_prompt"],
        self_rating=cfg.get("self_rating", False)))
    end = {"event": "end", **state}
    if usage_before is not None:
        # OpenRouter's key total lags the calls, sometimes by minutes: poll until it has caught up
        # with our own sum (within 5%) or until we have waited 4 minutes
        deadline = time.monotonic() + 240
        actual = key_usage() - usage_before
        while actual < 0.95 * state["spent"] and time.monotonic() < deadline:
            time.sleep(15)
            actual = key_usage() - usage_before
        end.update(key_usage_delta_usd=round(actual, 6),
                   matches_reported=abs(actual - state["spent"]) <= max(0.01, 0.05 * actual))
        print(f"OpenRouter key spend this run: ${actual:.4f}   (our per-call sum: ${state['spent']:.4f})")
        if not end["matches_reported"]:
            print("NOTE: these differ. If OpenRouter's number is LOWER its total is probably still lagging; "
                  "re-check the key's usage later. If it is HIGHER, other calls used the key meanwhile, "
                  "or a provider under-reported per-call cost: investigate before the next run.")
    log_run(run_log, end)
    print(f"results: {out}\nrun log: {run_log}")


if __name__ == "__main__":
    main()
