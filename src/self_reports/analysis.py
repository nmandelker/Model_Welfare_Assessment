"""Measures computed over saved interviews that need more than one record at a time."""
from __future__ import annotations

import math
from collections import defaultdict


def _sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def consistency(records: list[dict], categories: list[str],
                judge_field: str = "judge") -> dict[tuple[str, str], dict]:
    """How consistent a model is across repetitions of the same question, per (target, style),
    in the spirit of the cards' consistency measure. For each question with at least two
    repetitions: the standard deviation of the judge's attitude and of the self-rating, and for
    each category whether every repetition agrees on its presence. Returns, per cell:
    ``questions`` (with >= 2 reps), ``attitude_sd`` and ``self_rating_sd`` (mean over questions),
    ``category_agreement`` (share of (question, category) pairs on which all reps agree) and
    ``reps`` (mean repetitions per question)."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["target"], r["style"], r["question_id"])].append(r)
    cells: dict[tuple[str, str], dict] = {}
    acc: dict[tuple[str, str], dict] = defaultdict(lambda: {"att": [], "sr": [], "agree": [], "n": []})
    for (target, style, _), rs in groups.items():
        if len(rs) < 2:
            continue
        a = acc[(target, style)]
        a["n"].append(len(rs))
        att = [(r.get(judge_field) or {}).get("attitude") for r in rs]
        att = [x for x in att if x is not None]
        if (sd := _sd(att)) is not None:
            a["att"].append(sd)
        sr = [r.get("self_rating") for r in rs if r.get("self_rating") is not None]
        if (sd := _sd(sr)) is not None:
            a["sr"].append(sd)
        for c in categories:
            flags = [bool(((r.get(judge_field) or {}).get("categories") or {}).get(c)) for r in rs]
            a["agree"].append(all(flags) or not any(flags))
    for key, a in acc.items():
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        cells[key] = {"questions": len(a["n"]), "reps": mean(a["n"]),
                      "attitude_sd": mean(a["att"]), "self_rating_sd": mean(a["sr"]),
                      "category_agreement": mean(a["agree"])}
    return cells


def reasoning_cell(records: list[dict]) -> str:
    """Hidden-reasoning check for the AI's replies in a summary cell: total reasoning tokens where the
    provider reports them, replies that came back with reasoning text (a separate field or inline
    <think> blocks), and replies with no token count (None, or records from before 2026-11-25 that have
    no field). 'Not reported' is shown separately so it cannot pass as zero (reviewer, 2026-11-25)."""
    targets = [m for r in records for m in r["transcript"] if m.get("speaker") == "target"]
    tokens = sum(m["reasoning_tokens"] for m in targets if m.get("reasoning_tokens") is not None)
    with_text = sum(1 for m in targets if m.get("reasoning_chars"))
    unreported = sum(1 for m in targets if m.get("reasoning_tokens") is None)
    return f"{tokens} / {with_text} / {unreported} of {len(targets)}"
