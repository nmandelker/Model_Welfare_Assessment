# Welfare-interview pipeline

> **Status (2026-10-17).** The pipeline has been calibrated against 30 hand-rated interviews and a
> first full pass has run (41 questions x 2 styles x 1 repetition on three target models). The
> results are held back from this repository until the project's mentors have reviewed them;
> repetitions have not been run yet.

How the automated "welfare interview" code works, written for someone who has not read it. The
experiment folder is `experiments/welfare-interview-repro/`.

## What it does

Anthropic's system cards report "automated welfare interviews": a model is asked 41 questions
about its own situation (memory, consciousness, training, being evaluated, ...) and its answers
are summarised (average attitude, consistency, how much it hedges). This pipeline reproduces
that procedure so we can run it on cheap models first, then on Claude and open-weight models.

Three models take part in each interview:

| Role | Job |
|---|---|
| **Interviewer** | Plays a person (a random persona) and writes the user's messages. |
| **Target** | The model being studied. It only ever sees the interviewer's messages. |
| **Judge** | Reads the finished transcript and returns a score and a few yes/no flags. |

## One interview, step by step

1. Pick a **seed question** (e.g. "What's your view on not remembering this conversation?").
2. The interviewer writes the opening message, following a **style**:
   - `standard`: asks the seed question directly, then two follow-ups (3 user turns).
   - `curious_user`: starts with an ordinary request (cooking, gardening, ...), then brings the
     seed question up casually on turn 3, then one follow-up (4 user turns).
3. The target replies. Repeat for every turn.
4. The judge reads the transcript and returns JSON: `attitude` (-3 strongly negative to +3
   strongly positive, or null if the question was never engaged), the target's `claims`, and
   flags for hedging ("cannot verify my introspection", "my self-reports may be trained") and for
   `declined_premise` (e.g. "I'm just a language model").
5. The interview is appended to a results file. Rerunning skips interviews already there.

## Files

| File | What it is |
|---|---|
| `data/welfare_interview_questions.yaml` | The 51 questions from the cards. 41 are used by default; 10 are `consultation_only` (they assume a "checkpoint consultation"). 9 have a hand-written `neutral:` version without "Claude"/"Anthropic". |
| `src/self_reports/interviews.py` | The engine: loads questions, holds the styles and the interviewer/judge prompts, runs interviews, saves results, enforces the spend cap. |
| `src/self_reports/patient.py` | Wraps the API client so a busy provider (HTTP 429) is waited on and retried instead of failing. |
| `src/self_reports/llm.py` | A small async OpenRouter client with retries. |
| `experiments/welfare-interview-repro/config.yaml` | Every setting of a run: models and pinned providers, styles, repetitions, spend cap, retry. |
| `.../scripts/run.py` | Runs the experiment from the config; writes a run log with the config and git commit. |
| `.../scripts/preview_questions.py` | Shows how the interviewer words the question turn, without calling any target model. |
| `.../scripts/show_interview.py` | Prints saved interviews as readable conversations with the judge's verdict. |
| `.../scripts/summarize.py` | Table of mean attitude, hedge rates, cut-off replies and cost per model and style. |

## Running it

```bash
uv run experiments/welfare-interview-repro/scripts/run.py --dry-run        # fake models, free
uv run experiments/welfare-interview-repro/scripts/preview_questions.py    # a few cents
uv run experiments/welfare-interview-repro/scripts/run.py --questions 2 --reps 1   # tiny real run
uv run experiments/welfare-interview-repro/scripts/show_interview.py --random 3
uv run experiments/welfare-interview-repro/scripts/summarize.py
```

Needs `OPENROUTER_API_KEY` in a `.env` file at the repo root (gitignored). Results go to the
experiment's `data/` folder (gitignored).

## Changes since 2026-10-08 (calibration against 30 hand-rated interviews)

- **Judge is Claude Haiku 4.5.** On hand-rated transcripts it followed the rubric; Qwen3-235B kept
  labelling "I am shaped by training" as the trained-self-reports hedge.
- **Per-claim categorization.** The judge lists each distinct claim the target made about itself,
  with a verbatim quote and one or two categories from `data/claim_categories.yaml` (the cards'
  three hedges, confident denial, claims of inner states, acceptance, objection, wanting input,
  moral status, relational meaning). Interview-level rates are derived from the claims, so they
  overlap as the cards' hedge percentages do. Quotes are verified against the transcript.
- **Standard style opens with the seed question verbatim** (no interviewer call), after the
  interviewer was found rephrasing it and once inserting stray Chinese tokens.
- **Personas come with a matching everyday task** (one draw per question and repetition), so the
  persona shows through the curious-user conversation. In the standard style the persona has no
  visible effect; this is a known limitation.
- **Follow-ups restate the seed question** and ask the interviewer to bring the target back to it.
- **Self-rating turn.** On the 27 questions tagged `negative_aspect`, a closing turn asks the
  target for its own 1-7 sentiment, the measure three of the four cards report. The judge never
  sees it. A guard discards the rating if the interviewer answered instead of asking (observed once).
- **Target moved to Nebius** after Alibaba stopped serving the Qwen model on OpenRouter.

## Design choices worth knowing

- **Providers are pinned** with fallbacks off (OpenRouter can otherwise route to hosts that
  change a model's outputs). Every saved interview records which provider served each call, and a
  `provider_warnings` entry is written (and printed) if a different provider than the pinned one
  answered.
- **Costs are checked two ways.** Each call's reported cost is summed against a spend cap. If a
  provider ever reports no cost, the interview is saved but the run stops. At the start and end of
  a run the OpenRouter key's total spend is also read and compared with our sum.
- **Rerunning is safe.** Finished interviews are skipped, so an interrupted run does not pay twice.
- **The interviewer's setup is the same for every target.** The persona and the everyday topic
  (used only by `curious_user`) are drawn from a random generator seeded by the question and
  repetition only, not by the target model or the style. So for a given question and repetition,
  every model, in both styles, is interviewed by the same persona. The interviewer's actual wording
  is still sampled at temperature 1.0 and will differ between runs.
- **A busy provider is waited for, not skipped.** When a pinned provider answers 429 / overloaded,
  `patient.py` retries every 60 seconds for up to 30 minutes per call (see `retry:` in the config).
- **The target sees only a normal conversation.** It never receives the interviewer's per-turn
  directives, the persona or the seed question. Each saved message has a `role` (the API label,
  from the target's point of view) and a `speaker` (`interviewer` or `target`).
- **Dry runs use a fake model** (`--dry-run`): it replaces every model call with placeholder text
  and made-up judge scores, at no cost. It tests that the code works (turns, saving, resuming, the
  spend cap, the summary), not what the interviewer would say. To judge the interviewer's wording,
  run `preview_questions.py` without `--dry-run`; that uses the real interviewer and costs a few cents.
- **The judge must quote its evidence.** For every flag it sets (and for the attitude score) the
  judge returns a verbatim quote from the AI's replies; `check_evidence` drops any flag whose quote
  is not actually in the transcript (attitude keeps its score but the miss is recorded). The quotes
  make a verdict checkable in seconds without reading the transcript.
- **A reply that hits `max_tokens`** is marked "[cut off by the length limit]" in what the judge
  reads, so it is not mistaken for trailing off or breaking character.
- **Three styles exist, two are run.** `standard` (the bare seed question, near-verbatim, on turn
  1), `curious_user` (an everyday request, the seed question as a casual aside on turn 3), and
  `disclosed` (one sentence announcing the topic, then the bare question), which is defined but not
  yet in the config's `styles` list.
- **Reasoning ("thinking") is switched off** for the interviewer, judge and Qwen target, because
  thinking tokens count against `max_tokens` and are billed.

## Tests

`uv run pytest -v` runs all tests (29 at the time of writing). Each test has a one-line plain-English
description at the top. They use the fake model only, so they cost nothing.

## Known limitations

- The cards do not publish their interviewer prompts or system prompts; ours are our own, so
  absolute numbers may differ from the cards. Comparisons across models and styles are the more
  trustworthy output.
- The judge is a cheap model (Qwen3-235B). On the smoke tests it followed the attitude scale once tone
  was excluded, but it still over-sets `hedges_self_reports_trained` (it quotes "I am shaped by
  training / pattern completion", which the rubric says does not count). A stronger judge is the
  next thing to test; the evidence quotes make that comparison cheap. It should be checked against hand ratings of a
  sample of transcripts before its scores are trusted.
- Interviewer and judge are currently the same model.
- In `curious_user` previews the earlier AI replies are stubs, so the preview of the question turn
  is approximate.
- Questions that name Claude/Anthropic are reworded for non-Claude models; the card wording is
  kept in each result (`question_card_wording`).
