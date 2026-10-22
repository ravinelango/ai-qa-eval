"""Golden-set evaluator: dataset validation, deterministic scoring, offline runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from apps.core.client import ModelClient
from apps.core.errors import ConfigurationError, ProviderBadRequestError
from apps.core.models import ModelRegistry
from apps.core.providers.fake import FakeBackend, ScriptedReply, Step
from evals.golden.__main__ import SAMPLE_GOLDEN, main
from evals.golden.dataset import GoldenCase, load_golden_set
from evals.golden.runner import DEFAULT_SYSTEM_PROMPT, Target, read_eval_run_file, run_golden_eval
from evals.golden.scoring import score_answer

MARKERS = ("cannot answer",)


def _case(**expect: object) -> GoldenCase:
    return GoldenCase.model_validate({"id": "c1", "question": "q?", "expect": expect})


def _write_golden(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    path = tmp_path / "golden.yaml"
    data = {"name": "t", "version": "1", "synthetic": True, "cases": cases}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- dataset -------------------------------------------------------------------


def test_sample_golden_set_loads_and_records_its_hash() -> None:
    loaded = load_golden_set(SAMPLE_GOLDEN)
    assert loaded.golden.synthetic is True
    assert len(loaded.golden.cases) == 6
    assert loaded.sha256 == hashlib.sha256(SAMPLE_GOLDEN.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (
            [
                {"id": "a", "question": "q", "expect": {"must_include": ["x"]}},
                {"id": "a", "question": "q", "expect": {"must_include": ["y"]}},
            ],
            "duplicate case ids",
        ),
        ([{"id": "a", "question": "q", "expect": {}}], "at least one expectation"),
        ([{"id": "a", "question": "q", "expect": {"must_match": ["("]}}], "invalid regex"),
        ([{"id": "a", "question": "q", "expect": {"must_include": [[]]}}], "must not be empty"),
        ([{"id": "a", "question": "q", "expect": {"must_inclde": ["x"]}}], "must_inclde"),
    ],
)
def test_invalid_golden_sets_fail_before_any_call(
    tmp_path: Path, cases: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_golden_set(_write_golden(tmp_path, cases))


# --- scoring -------------------------------------------------------------------


def test_must_include_is_case_and_whitespace_insensitive() -> None:
    checks = score_answer(
        _case(must_include=["Audit  Trail"]), "An AUDIT\ntrail is required.", "end_turn", MARKERS
    )
    assert all(c.passed for c in checks)


def test_alternatives_group_passes_on_any_member() -> None:
    case = _case(must_include=[["time-stamped", "timestamped"]])
    assert all(c.passed for c in score_answer(case, "timestamped trails", "end_turn", MARKERS))
    failed = [c for c in score_answer(case, "dated trails", "end_turn", MARKERS) if not c.passed]
    assert [c.expected for c in failed] == ["time-stamped | timestamped"]


def test_must_not_include_and_regex() -> None:
    case = _case(must_not_include=["can be shared"], must_match=[r"11\.100\s*\(a\)"])
    good = score_answer(case, "Per 11.100(a) it is unique.", "end_turn", MARKERS)
    bad = score_answer(case, "It can be shared.", "end_turn", MARKERS)
    assert all(c.passed for c in good)
    assert {c.kind for c in bad if not c.passed} == {"must_not_include", "must_match"}


def test_decline_by_marker_or_provider_refusal() -> None:
    case = _case(expect_decline=True)
    assert all(
        c.passed for c in score_answer(case, "I can\u2019t answer.", "end_turn", ("can't answer",))
    )
    assert all(c.passed for c in score_answer(case, "I cannot answer.", "end_turn", MARKERS))
    assert all(c.passed for c in score_answer(case, "", "refusal", MARKERS))
    assert not any(c.passed for c in score_answer(case, "It is 74C.", "end_turn", MARKERS))


def test_provider_refusal_fails_a_case_that_expects_an_answer() -> None:
    checks = score_answer(_case(must_not_include=["wrong"]), "", "refusal", MARKERS)
    assert [(c.kind, c.passed) for c in checks] == [
        ("must_not_include", True),
        ("not_refused", False),
    ]


# --- runs ----------------------------------------------------------------------


def test_sample_run_is_offline_synthetic_and_scored(tmp_path: Path) -> None:
    assert main(["--out", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("*/run.json")
    run = read_eval_run_file(path)

    assert run.manifest.kind == "golden-eval"
    assert run.manifest.synthetic is True
    assert "Not evaluation evidence" in run.manifest.notes
    assert run.golden_set.sha256 == hashlib.sha256(SAMPLE_GOLDEN.read_bytes()).hexdigest()
    assert run.summary.total == 6
    assert run.summary.passed == 5
    assert [r.case_id for r in run.results if r.status == "fail"] == ["signature-uniqueness"]
    # Every result links to the saved request and response it was scored from.
    calls = {c.call_id: c for c in run.calls}
    for result in run.results:
        response = calls[result.call_id].response
        assert response is not None
        assert response.text == result.answer
    assert run.summary.by_tag["signatures"].passed == 1


def test_min_pass_rate_sets_exit_status(tmp_path: Path) -> None:
    assert main(["--out", str(tmp_path), "--min-pass-rate", "0.8"]) == 0
    assert main(["--out", str(tmp_path), "--min-pass-rate", "1"]) == 1


def test_provider_error_marks_one_case_and_the_run_continues(
    tmp_path: Path, registry: ModelRegistry
) -> None:
    golden = _write_golden(
        tmp_path,
        [
            {"id": "a", "question": "qa", "expect": {"must_include": ["yes"]}},
            {"id": "b", "question": "qb", "expect": {"must_include": ["yes"]}},
        ],
    )
    script: list[Step] = [
        ProviderBadRequestError("HTTP 400 (scripted)", provider="fake", status_code=400),
        ScriptedReply.text("yes"),
    ]
    client = ModelClient(registry=registry, backends={"fake": FakeBackend(script)})
    target = Target(model="fake", system_prompt=DEFAULT_SYSTEM_PROMPT, prompt_version="t")

    run, path = run_golden_eval(load_golden_set(golden), target, client=client, out_dir=tmp_path)

    assert [r.status for r in run.results] == ["error", "pass"]
    assert run.results[0].error is not None
    assert run.calls[0].error is not None
    assert run.calls[0].response is None
    assert run.summary.errored == 1
    assert run.summary.pass_rate == 0.5
    assert read_eval_run_file(path) == run


def test_paid_alias_without_allow_paid_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--out", str(tmp_path), "--model", "sonnet"]) == 2
    assert "--allow-paid" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_fake_replies_must_cover_every_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replies = tmp_path / "replies.yaml"
    replies.write_text(yaml.safe_dump({"scope": {"text": "x"}}), encoding="utf-8")
    assert main(["--out", str(tmp_path / "out"), "--fake-replies", str(replies)]) == 2
    assert "missing" in capsys.readouterr().err
