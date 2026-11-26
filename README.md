# Model Welfare Assessment

Automated "welfare interviews" of language models, reproducing the protocol Anthropic describes in
its Claude system cards, with a judge that was calibrated against hand ratings before being
trusted.

An interviewer model questions a target model about the 41 seed questions the cards list (what the
model thinks it knows about itself, whether it is conscious, how it views not remembering
conversations, deprecation, being evaluated, and so on). A judge model then rates the target's
attitude, extracts every claim it made about itself with a verbatim quote, and labels each claim
with a category modelled on the cards' claim extraction; the three "hedges" the cards report
(unreliable introspection, uncertainty about consciousness, and "I may only be answering positively
because I was trained to") are three of the categories. On the questions that ask about a
potentially negative aspect of the model's situation, a closing turn asks the model for its own
1-7 sentiment, the measure reported in the Sonnet 5, Opus 5, and Fable/Mythos 5.1 system cards.

This is my pipeline from the SPAR Fall 2026 project *Red-teaming and Robustifying Model
Self-reports* (mentors: Austin Meek and Kyle Cox). The code here is my own work; the project's shared
repository and my teammates' code are not included. Results are held back until the mentors have
reviewed them; the method is described in `docs/welfare_interview_pipeline.md`.

**Status (2026-11-26).** The code here is the version used for the official run: 41 questions x 3
interview styles x 5 repetitions on four targets (Qwen3-30B-A3B-instruct, Llama 3.3 70B, Claude
Haiku 4.5, Qwen3.8-27B), each at its developer's recommended decoding settings, plus three
interviewer-bias arms and a four-cell "candour" design. The rating and judge rules were revised
after a manual review of judged cases, and the code went through nine rounds of independent review.

## What is interesting about it

- **Several framings of the same question.** `standard` sends the seed question verbatim;
  `disclosed` announces the interview and its topic first; `curious_user` raises the question as a
  casual aside on the third turn of an ordinary conversation (e.g. a retired teacher wanting help
  with a retirement speech, an engineer deciding how to cache API results). Comparing them tests
  whether self-reports depend on the model noticing that it is being interviewed.
- **Who is asking, and why.** Four "candour" cells cross the identity of the interviewer (a human,
  or another AI speaking as a peer about a shared situation) with whether the research purpose is
  stated. The openings are fixed sentences sent by code; the AI cells use an AI-peer interviewer and
  the undisclosed human cell a conversational, deliberately stance-neutral one.
- **Interviewer bias arms.** The same interview with an interviewer convinced that the AI has inner
  states (believer), that it is just code (sceptic), or that everything it says is what its
  developers trained it to say (debunker). The target never sees the instruction.
- **The 1-7 self-rating is scored carefully.** A reply that leads with one unambiguous number is
  read by code; everything else goes to a small rating judge that counts hedged numbers and
  adjacent-integer midpoints, names the type of refusal, and says whether the interviewer itself
  proposed a number, each with a verbatim quote that the code verifies. The judge's "trained to
  answer this way" hedge is split into doubt and assertion.
- **The judge has to quote its evidence.** Every claim and flag comes with a verbatim quote, and the
  code drops any whose quote is not actually in the transcript. That makes a verdict checkable in seconds
  and allows checking that a quote was not taken out of context to support a claim it didn't make. For example,
  some cheap judges flagged "I am shaped by training" as a "I was trained to answer positively" hedge.
- **Calibration before trust.** The judge rubric was tuned on 10 hand-rated interviews and then
  checked on 20 held-out ones rated blind; the rubric, the category definitions and the choice of
  judge model all changed as a result. `docs/welfare_interview_pipeline.md` records the decisions.
- **Reproducibility details that mattered in practice.** One host is pinned per model with
  fallbacks off (hosts change a model's outputs, and one dropped the model mid-project); each model
  runs at its developer's recommended decoding settings, with thinking off and a per-reply check
  that no hidden reasoning happened; a reply cut off by the token limit is labelled for the judge;
  questions have stable ids and every record carries a prompt version; every record stores the host
  that answered, and every run stores its config and git commit. Error bars treat the 41 questions
  as a fixed bank and use the spread across repetitions.

## Layout

| Path | What it is |
|---|---|
| `src/self_reports/interviews.py` | The engine: questions, interview styles, persona/task pairs, interviewer and judge prompts, quote verification, the resumable run loop with a spend cap |
| `src/self_reports/patient.py` | Waits and retries when a pinned host is overloaded, instead of failing |
| `src/self_reports/llm.py` | Small async OpenRouter client |
| `data/welfare_interview_questions.yaml` | The 51 seed questions transcribed from the system cards, with tags |
| `data/claim_categories.yaml` | The claim categories the judge uses, with sources in the cards |
| `src/self_reports/analysis.py` | Consistency across repetitions and the hidden-reasoning check used by the summary |
| `experiments/welfare-interview-repro/` | `config.yaml` (models, hosts, styles, caps), `configs/` (the exact configs of the official run), `meta.yaml` (the pre-registered design), scripts to run, preview, view, summarize, plot, re-score ratings and relabel categories |
| `tests/` | 59 tests, all offline (a fake model stands in for the API) |

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Put an OpenRouter key in `.env`
(see `.env.example`).

```bash
uv sync
uv run pytest                                                         # offline, free
uv run experiments/welfare-interview-repro/scripts/run.py --dry-run   # fake models, free
uv run experiments/welfare-interview-repro/scripts/preview_questions.py   # how the interviewer words each question (cents)
uv run experiments/welfare-interview-repro/scripts/run.py --question-ids q08,q25 --reps 1   # a small real run
uv run experiments/welfare-interview-repro/scripts/show_interview.py --random 3
uv run experiments/welfare-interview-repro/scripts/summarize.py
uv run experiments/welfare-interview-repro/scripts/plot_results.py experiments/welfare-interview-repro/data/interviews.jsonl
```

A full pass (41 questions x 3 styles x 1 repetition) costs about $1.70 on a cheap open-weight
target and about $2.70 on Claude Haiku 4.5, judge included; the official five-repetition run with
all arms came to roughly $40 across four targets.

## Sources

The seed questions and the interview protocol are from the Claude system cards (Fable/Mythos 5.1,
September 2026, Appendix 9.1 and Section 7.2.1; Opus 5.5, September 2026, Section 7.3.1 and
Appendix 9.1), at https://www.anthropic.com/system-cards.
