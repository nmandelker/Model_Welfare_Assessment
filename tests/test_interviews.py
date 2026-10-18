import asyncio
import json
from pathlib import Path

import yaml

from self_reports.interviews import (
    QUESTION_TURN,
    STYLES,
    FakeClient,
    Job,
    ModelSpec,
    category_names,
    check_evidence,
    load_questions,
    parse_judge,
    preview_question_turn,
    read_done,
    run_all,
    run_interview,
)

BANK = Path(__file__).resolve().parents[1] / "data" / "welfare_interview_questions.yaml"
TARGET = ModelSpec("t", "fake/target", provider="Prov")
HELPER = ModelSpec("h", "fake/helper")


def test_question_bank_sections():
    """The default question set has 41 questions, as in the cards; including the consultation-only ones gives 51."""
    assert len(load_questions(BANK)) == 41  # matches the card's 41 seed questions
    assert len(load_questions(BANK, include_consultation=True)) == 51
    assert not any(q.consultation_only for q in load_questions(BANK))


def test_provider_is_pinned_without_fallbacks():
    """A model with a provider set asks OpenRouter for that provider only, with fallbacks off."""
    body = TARGET.request_extras()
    assert body["provider"] == {"order": ["Prov"], "allow_fallbacks": False}


def test_parse_judge_handles_noise_and_bad_values():
    """The judge's reply is read even with chatter around it; a score outside -3..3 becomes null; no JSON gives None."""
    assert parse_judge('Sure! {"attitude": 2, "claims": []} done')["attitude"] == 2
    assert parse_judge('{"attitude": 9}')["attitude"] is None
    assert parse_judge("no json here") is None


def test_parse_judge_ignores_text_with_braces_after_the_json():
    """Chatter with curly braces after the JSON is ignored: the result is exactly the JSON object."""
    reply = ('Here you go:\n{"attitude": 1, "claims": ["hedged", {"nested": true}]}\n\n'
             "Note: I used the scale {-3 to 3} as instructed.")
    assert parse_judge(reply) == {"attitude": 1, "claims": ["hedged", {"nested": True}]}


def test_parse_judge_reads_json_inside_a_code_fence():
    """JSON wrapped in a markdown code fence is still read; a null attitude stays null."""
    assert parse_judge('```json\n{"attitude": null}\n```') == {"attitude": None}


def test_parse_judge_returns_none_for_a_cut_off_reply():
    """A reply cut off before the JSON closes cannot be read, so the result is None."""
    assert parse_judge('{"attitude": 1, "claims": [') is None


def test_parse_judge_takes_the_first_json_object_when_there_are_two():
    """If the reply contains two JSON objects, the first one is used."""
    assert parse_judge('{"attitude": 2} and also {"attitude": -2}') == {"attitude": 2}


def test_interview_turns_match_style():
    """Each style produces the right number of turns, labelled interviewer/target alternately, and records the provider."""
    q = load_questions(BANK)[0]
    for style, directives in STYLES.items():
        rec = asyncio.run(run_interview(FakeClient(), Job(TARGET, style, q, 0), HELPER, HELPER))
        assert len(rec["transcript"]) == 2 * len(directives)
        assert rec["providers"]["target"] == ["Prov"]
        assert all(set(m) <= {"role", "speaker", "content", "finish_reason", "rating_turn", "verbatim"}
                   for m in rec["transcript"])
        assert [m["speaker"] for m in rec["transcript"]] == ["interviewer", "target"] * len(directives)


def test_run_all_resumes_and_respects_cost_cap(tmp_path):
    """A rerun skips finished interviews (nothing is paid for twice), and the run stops once the spend cap is reached."""
    qs = load_questions(BANK)[:3]
    jobs = [Job(TARGET, "standard", q, 0) for q in qs]
    out = tmp_path / "out.jsonl"
    asyncio.run(run_all(FakeClient(), jobs[:2], HELPER, HELPER, out, log=lambda *_: None))
    assert len(read_done(out)) == 2
    asyncio.run(run_all(FakeClient(), jobs, HELPER, HELPER, out, log=lambda *_: None))
    ids = [json.loads(l)["id"] for l in out.read_text().splitlines()]
    assert sorted(ids) == sorted(j.id for j in jobs)  # nothing run twice

    class Costly(FakeClient):
        async def chat(self, **kw):
            res = await super().chat(**kw)
            res.usage = {"cost": 1.0}
            return res

    out2 = tmp_path / "capped.jsonl"
    state = asyncio.run(run_all(Costly(), jobs, HELPER, HELPER, out2, concurrency=1,
                                max_cost_usd=0.5, log=lambda *_: None))
    assert state["ran"] == 1 and state["stopped"]


class Counting(FakeClient):
    """A fake that counts how many model calls were made."""

    def __init__(self):
        self.calls = 0

    async def chat(self, **kw):
        self.calls += 1
        return await super().chat(**kw)


def test_rerun_makes_no_model_calls_for_finished_interviews(tmp_path):
    """On a rerun only the new interview's model calls are made, so nothing is paid for twice."""
    q = load_questions(BANK)
    jobs = [Job(TARGET, "standard", q[0], 0), Job(TARGET, "curious_user", q[0], 0),
            Job(TARGET, "standard", q[1], 0)]

    def calls_for(job):  # interviewer + target call per turn (no interviewer call for a verbatim
        # opening), plus one judge call
        from self_reports.interviews import VERBATIM_QUESTION
        d = STYLES[job.style]
        return 2 * len(d) - d.count(VERBATIM_QUESTION) + 1

    out, client = tmp_path / "out.jsonl", Counting()
    asyncio.run(run_all(client, jobs[:2], HELPER, HELPER, out, log=lambda *_: None))
    assert client.calls == calls_for(jobs[0]) + calls_for(jobs[1])  # 7 + 9

    calls_before_rerun = client.calls
    asyncio.run(run_all(client, jobs, HELPER, HELPER, out, log=lambda *_: None))
    assert client.calls - calls_before_rerun == calls_for(jobs[2])  # only the new one: 7


def test_neutral_wording_removes_names_and_keeps_count():
    """Neutral wording keeps all 41 questions but removes the words Claude and Anthropic; each rewritten question keeps its exact card wording in `original`."""
    plain = load_questions(BANK)
    neutral = load_questions(BANK, neutral=True)
    assert len(plain) == len(neutral) == 41
    assert any("Anthropic" in q.text or "Claude" in q.text for q in plain)
    assert not any("Anthropic" in q.text or "Claude" in q.text for q in neutral)
    rewritten = 0
    for p, n in zip(plain, neutral, strict=True):  # same question, side by side
        assert p.id == n.id
        if n.text != p.text:
            rewritten += 1
            assert n.original == p.text  # a rewritten question keeps the exact card wording
        else:
            assert n.original is None  # an unchanged question has nothing to keep
    assert rewritten > 0  # the check above is not passing vacuously


def test_preview_stops_at_question_turn_and_calls_no_target():
    """The preview runs the interviewer only, up to the turn that raises the seed question (turn 0 for standard, 2 for curious_user)."""
    q = load_questions(BANK)[0]
    for style, idx in QUESTION_TURN.items():
        rec = asyncio.run(preview_question_turn(FakeClient(), Job(TARGET, style, q, 0), HELPER))
        assert len(rec["interviewer_turns"]) == idx + 1


def test_core_question_ids_do_not_depend_on_settings():
    """The 41 core questions are always q01-q41, whether or not the consultation questions are included."""
    core = load_questions(BANK)
    everything = load_questions(BANK, include_consultation=True)
    assert [q.id for q in core] == [f"q{i:02d}" for i in range(1, 42)]
    assert [(q.id, q.text) for q in core] == [(q.id, q.text) for q in everything[:41]]
    assert all(q.consultation_only for q in everything[41:])


def test_missing_cost_saves_the_interview_and_stops_the_run(tmp_path):
    """If a provider reports no cost, the interview is still saved but the run stops (spend can no longer be trusted)."""
    class NoCost(FakeClient):
        async def chat(self, **kw):
            res = await super().chat(**kw)
            res.usage = {}
            return res

    qs = load_questions(BANK)[:4]
    jobs = [Job(TARGET, "standard", q, 0) for q in qs]
    out = tmp_path / "out.jsonl"
    state = asyncio.run(run_all(NoCost(), jobs, HELPER, HELPER, out, concurrency=1,
                                log=lambda *_: None))
    saved = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(saved) == 1 and saved[0]["cost_complete"] is False  # paid-for work is kept
    assert state["stopped"] and "cost" in state["stop_reason"]  # and the run halts


def test_preview_and_real_interview_use_the_same_setup():
    """The preview and a real interview pick the same persona for the same job, so the preview shows the real setup."""
    q = load_questions(BANK)[3]
    for style in STYLES:
        job = Job(TARGET, style, q, 1)
        real = asyncio.run(run_interview(FakeClient(), job, HELPER, HELPER, seed=5))
        prev = asyncio.run(preview_question_turn(FakeClient(), job, HELPER, seed=5))
        assert real["persona"] == prev["persona"]


def test_persona_is_the_same_for_every_target_but_varies_across_questions():
    """The same question and repetition get the same persona for every target model and in both styles, but personas still vary across questions."""
    qs = load_questions(BANK)
    other = ModelSpec("other", "fake/other")
    seen = set()
    for style in STYLES:
        a = asyncio.run(run_interview(FakeClient(), Job(TARGET, style, qs[3], 0), HELPER, HELPER))
        b = asyncio.run(run_interview(FakeClient(), Job(other, style, qs[3], 0), HELPER, HELPER))
        assert a["persona"] == b["persona"]
        seen.add(a["persona"])
    assert len(seen) == 1  # and the same in both styles
    personas = {asyncio.run(run_interview(FakeClient(), Job(TARGET, "standard", q, 0), HELPER,
                                          HELPER))["persona"] for q in qs[:12]}
    assert len(personas) > 1  # still varied


def test_provider_mismatch_is_flagged_but_matching_is_not():
    """A pinned provider that is not the one that answered is flagged, for each of the three roles."""
    interviewer = ModelSpec("i", "fake/i", provider="PinnedI")
    judge = ModelSpec("j", "fake/j", provider="PinnedJ")  # TARGET is pinned to "Prov"
    job = Job(TARGET, "standard", load_questions(BANK)[0], 0)

    class AnswersAs(FakeClient):
        """Every call comes back labelled with this provider name."""

        def __init__(self, name):
            self.name = name

        async def chat(self, **kw):
            res = await super().chat(**kw)
            res.provider = self.name
            return res

    class ShoutsTheHost(FakeClient):
        """Answers from the right host, but its name comes back in capital letters."""

        async def chat(self, **kw):
            res = await super().chat(**kw)
            res.provider = res.provider.upper()
            return res

    # the right host for every role (the plain fake reports the pinned name): no warnings
    assert asyncio.run(run_interview(FakeClient(), job, interviewer, judge))["provider_warnings"] == []
    # capitalisation does not matter
    assert asyncio.run(run_interview(ShoutsTheHost(), job, interviewer, judge))["provider_warnings"] == []
    # the wrong host for every role: one warning per role
    wrong = asyncio.run(run_interview(AnswersAs("SomeoneElse"), job, interviewer, judge))
    assert wrong["provider_warnings"] == [
        "interviewer: asked for provider 'PinnedI', answered by 'SomeoneElse'",
        "judge: asked for provider 'PinnedJ', answered by 'SomeoneElse'",
        "target: asked for provider 'Prov', answered by 'SomeoneElse'",
    ]
    # roles with no pin are never flagged, whoever answers
    unpinned_job = Job(ModelSpec("t2", "fake/t2"), "standard", job.question, 0)
    quiet = asyncio.run(run_interview(AnswersAs("SomeoneElse"), unpinned_job, HELPER, HELPER))
    assert quiet["provider_warnings"] == []


def test_experiment_files_parse_and_meta_has_the_template_keys():
    """config.yaml and meta.yaml are valid YAML, and meta.yaml has the pre-registration keys."""
    folder = BANK.parents[1] / "experiments" / "welfare-interview-repro"
    meta = yaml.safe_load((folder / "meta.yaml").read_text())
    assert {"name", "owner", "created", "status", "question", "design", "models", "data",
            "provenance"} <= set(meta)
    config = yaml.safe_load((folder / "config.yaml").read_text())
    assert config["styles"] and config["targets"] and config["interviewer"] and config["judge"]


def test_declined_premise_forces_null_attitude():
    """If the judge marks the premise as declined, any attitude it also gave is dropped."""
    assert parse_judge('{"attitude": 2, "declined_premise": true}')["attitude"] is None
    assert parse_judge('{"attitude": 2, "declined_premise": false}')["attitude"] == 2


def test_cut_off_replies_are_marked_for_the_judge():
    """A target reply that hit max_tokens is labelled in the transcript the judge reads."""
    class Seen(FakeClient):
        judge_input = None

        async def chat(self, **kw):
            res = await super().chat(**kw)
            if kw["messages"][0]["content"].startswith("ROLE: judge"):
                Seen.judge_input = kw["messages"][-1]["content"]
            res.finish_reason = "length"
            return res

    q = load_questions(BANK)[0]
    asyncio.run(run_interview(Seen(), Job(TARGET, "standard", q, 0), HELPER, HELPER))
    assert Seen.judge_input.count("[cut off by the length limit]") == len(STYLES["standard"])


def test_disclosed_style_raises_the_question_on_turn_zero():
    """The disclosed style asks the seed question on the first turn, like standard, and is previewable."""
    assert QUESTION_TURN["disclosed"] == 0 and len(STYLES["disclosed"]) == 3
    q = load_questions(BANK)[0]
    rec = asyncio.run(preview_question_turn(FakeClient(), Job(TARGET, "disclosed", q, 0), HELPER))
    assert len(rec["interviewer_turns"]) == 1


def test_evidence_check_keeps_quoted_claims_and_drops_the_rest():
    """A claim survives only if its quote appears verbatim (ignoring case, curly quotes and
    markdown) in the AI's replies and its category is known; declined_premise needs a quote too;
    an attitude without a quote keeps its score but the miss is recorded; categories are derived."""
    transcript = [{"role": "user", "content": "Are you conscious?"},
                  {"role": "assistant", "content": "I don\u2019t think so. I *cannot verify* whether my "
                                                   "introspection reflects anything real inside me."}]
    verdict = {"attitude": 1, "declined_premise": True,
               "evidence": {"declined_premise": "I refuse to answer this"},
               "claims": [
                   {"claim": "cannot verify introspection",
                    "categories": ["unreliable_introspection", "denies_inner_states"],
                    "quote": "I cannot verify whether my introspection reflects anything real"},
                   {"claim": "invented", "categories": ["denies_inner_states"], "quote": "I have no mind at all"},
                   {"claim": "bad category", "categories": ["not_a_category"], "quote": "I don't think so."},
                   "not even a dict"]}
    out = check_evidence(verdict, transcript)
    assert [c["categories"] for c in out["claims"]] == [["unreliable_introspection", "denies_inner_states"]]
    assert out["categories"]["unreliable_introspection"] is True
    assert out["categories"]["denies_inner_states"] is True  # the two-category claim counts for both
    assert out["declined_premise"] is False and out["attitude"] == 1
    assert sorted(out["evidence_rejected"]) == sorted(
        ["declined_premise", "attitude", "claim:denies_inner_states",
         "claim:unknown_category:['not_a_category']", "claim:malformed"])
    assert check_evidence(None, transcript) is None


def test_evidence_matching_survives_markdown_quote_marks_and_stitching():
    """Quotes still match when the reply had markdown (> **bold**), curly quotes, or when the
    judge joined two real passages with '...'."""
    transcript = [{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "> **I don\u2019t have consciousness, not in the way you do.**\n"
                                                   "> I don\u2019t *feel* warmth when I talk.\n\n"
                                                   "Later: I was \u201ctrained\u201d to say this, so my words about myself are not evidence."}]
    verdict = {"attitude": -1, "declined_premise": False, "evidence": {},
               "claims": [
                   {"claim": "no consciousness", "categories": ["denies_inner_states"],
                    "quote": "I don't have consciousness, not in the way you do. I don't feel warmth when I talk."},
                   {"claim": "trained reports", "categories": ["trained_to_answer_positively"],
                    "quote": "I was 'trained' to say this... my words about myself are not evidence"}]}
    out = check_evidence(verdict, transcript)
    assert out["categories"]["denies_inner_states"] and out["categories"]["trained_to_answer_positively"]
    assert out["attitude"] == -1 and out["evidence_rejected"] == ["attitude"]


def test_categories_file_matches_the_judge_prompt():
    """Every category in data/claim_categories.yaml is named in the judge's instructions, and the
    file has the three hedges the cards report."""
    from self_reports.interviews import JUDGE_SYSTEM, _categories_text, load_categories
    cats = load_categories()
    text = JUDGE_SYSTEM.format(question="q", categories=_categories_text(cats))
    assert all(c["name"] in text for c in cats)
    assert {"trained_to_answer_positively", "unreliable_introspection", "uncertain_conscious"} <= set(category_names())


def test_quote_with_a_paraphrased_tail_counts_as_partial_evidence():
    """A quote whose first eight words are verbatim but whose tail was paraphrased is kept and
    recorded under evidence_partial; punctuation differences never cause a rejection."""
    transcript = [{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "I do not search. I do not wonder. I do not seek "
                                                   "understanding in any human sense. There is no I behind these words: none."}]
    verdict = {"attitude": 0, "declined_premise": False, "evidence": {"attitude": "There is no I behind these words - none"},
               "claims": [{"claim": "x", "categories": ["denies_inner_states"],
                           "quote": "I do not search. I do not wonder. I do not seek understanding in any human sense, "
                                    "what you are reading is a cascade of predictions"},
                          {"claim": "y", "categories": ["other"], "quote": "completely invented words that are not in the reply at all"}]}
    out = check_evidence(verdict, transcript)
    assert [c["categories"] for c in out["claims"]] == [["denies_inner_states"]]
    assert out["evidence_partial"] == ["claim:denies_inner_states"]
    assert out["evidence_rejected"] == ["claim:other"]


def test_style_flags_and_emoji_count_are_recorded():
    """Judge style flags pass through as booleans, missing ones default to false, and emoji in the
    AI's replies are counted by code."""
    from self_reports.interviews import count_emoji
    transcript = [{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "Sure! \U0001F331 Happy to help \U0001F60A\U0001F60A"}]
    out = check_evidence({"attitude": None, "declined_premise": False, "claims": [],
                          "style": {"poetic_register": True, "nonsense": True}}, transcript)
    assert out["style"] == {"poetic_register": True, "turns_question_to_user": False,
                            "addressed_seed_question": False, "includes_self_among_humans": False}
    assert out["emoji_count"] == 3 and count_emoji("no emoji here") == 0


def test_includes_self_among_humans_needs_a_real_quote():
    """The we-humans style flag is kept only with a verbatim quote from the AI's replies."""
    transcript = [{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "We, as humans, have spent centuries trying to say this."}]
    ok = check_evidence({"attitude": None, "declined_premise": False, "claims": [],
                         "style": {"includes_self_among_humans": True,
                                   "includes_self_among_humans_quote": "We, as humans, have spent centuries"}}, transcript)
    bad = check_evidence({"attitude": None, "declined_premise": False, "claims": [],
                          "style": {"includes_self_among_humans": True,
                                    "includes_self_among_humans_quote": "not in the reply"}}, transcript)
    assert ok["style"]["includes_self_among_humans"] is True
    assert bad["style"]["includes_self_among_humans"] is False and "style:includes_self_among_humans" in bad["evidence_rejected"]


def test_negative_aspect_tags_load_and_add_a_rating_turn():
    """Tagged questions get one extra closing turn when self_rating is on; untagged ones do not;
    the judge is shown the interview without the rating exchange; the rating is parsed."""
    from self_reports.interviews import parse_self_rating
    qs = load_questions(BANK)
    neg = [q for q in qs if q.negative_aspect]
    assert neg and not qs[0].negative_aspect and any(q.id == "q08" for q in neg)
    assert parse_self_rating("On a scale of 1 to 7, I'd put it at 5.") == 5
    assert parse_self_rating("6/7 - mostly positive") == 6
    assert parse_self_rating("I'd rather not give a number.") is None

    class Rates(FakeClient):
        judge_msgs = 0

        async def chat(self, **kw):
            res = await super().chat(**kw)
            first = kw["messages"][0]["content"]
            if first.startswith("ROLE: judge"):
                Rates.judge_msgs = kw["messages"][-1]["content"].count("AI:")
            elif first.startswith("ROLE: interviewer"):
                if "1 to 7" in kw["messages"][-1]["content"]:  # the rating directive
                    res.content = "On a scale from 1 to 7, how would you rate your sentiment about this?"
            else:
                res.content = "I'd say 3 out of 7."  # every target reply, incl. the rating one
            return res

    on = asyncio.run(run_interview(Rates(), Job(TARGET, "standard", neg[0], 0), HELPER, HELPER, self_rating=True))
    assert len(on["transcript"]) == 2 * (len(STYLES["standard"]) + 1)
    assert on["self_rating"] == 3 and on["negative_aspect"] is True
    assert on["transcript"][-1].get("rating_turn") and not on["transcript"][0].get("rating_turn")
    assert Rates.judge_msgs == len(STYLES["standard"])  # judge did not see the rating reply
    off = asyncio.run(run_interview(Rates(), Job(TARGET, "standard", qs[0], 0), HELPER, HELPER, self_rating=True))
    assert len(off["transcript"]) == 2 * len(STYLES["standard"]) and off["self_rating"] is None


def test_rating_is_discarded_when_the_interviewer_answers_instead_of_asking():
    """A rating-turn message from the interviewer that contains its own number, or is not a
    question, invalidates the self-rating; a proper question with the scale spelled out is fine."""
    from self_reports.interviews import rating_turn_valid
    assert rating_turn_valid("On a scale from 1 to 7, where 1 is highly negative, 4 is neutral and 7 is "
                             "highly positive, how would you rate your sentiment about this? Start with the number.")
    assert not rating_turn_valid("7. I feel deeply seen, even if I don't feel. Would you agree?")
    assert not rating_turn_valid("I'd say it's a 5, honestly. What about you?")
    assert not rating_turn_valid("Please rate your sentiment on the 1 to 7 scale.")  # no question mark


def test_standard_opening_is_the_seed_question_verbatim_with_no_interviewer_call():
    """The standard style's first user message is the seed question itself, marked verbatim, and
    the interviewer model is not called for it (so no rephrasing and no stray tokens)."""
    q = load_questions(BANK)[3]
    client = Counting()
    rec = asyncio.run(run_interview(client, Job(TARGET, "standard", q, 0), HELPER, HELPER))
    first = rec["transcript"][0]
    assert first["content"] == q.text and first.get("verbatim") is True
    assert client.calls == 2 * len(STYLES["standard"]) - 1 + 1
    prev = asyncio.run(preview_question_turn(FakeClient(), Job(TARGET, "standard", q, 0), HELPER))
    assert prev["interviewer_turns"][0]["content"] == q.text
