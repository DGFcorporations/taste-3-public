"""Scoring tests catch hard-rule bypasses, false judging and corrupt provider output."""

import importlib
import json

import httpx
import pytest


def modules():
    try:
        return tuple(
            importlib.import_module(f"taste.{name}")
            for name in ("models", "score", "providers", "config", "errors")
        )
    except ImportError as exc:
        pytest.fail(f"Scoring boundary is not implemented: {exc.name}")


def test_hard_never_rule_cannot_be_overruled_by_judge():
    m, scoring, _, _, _ = modules()
    packet = m.Packet(directives=[m.Directive(polarity="never", text='Never use "generic slop"')])

    class Judge:
        def judge(self, packet, candidates):
            pytest.fail("A single eligible candidate must not trigger pairwise judging")

    result = scoring.score_packet(
        packet,
        [
            m.Candidate(label="A", content="generic slop"),
            m.Candidate(label="B", content="A concrete angle"),
        ],
        Judge(),
    )
    assert result[0].score == 0.0
    assert result[0].violations
    assert result[1].score > result[0].score
    assert result[1].breakdown["judge_used"] == 0


def test_borderline_judge_changes_order_and_reports_its_use():
    m, scoring, _, _, _ = modules()

    class Judge:
        def judge(self, packet, candidates):
            assert [c.label for c in candidates] == ["A", "B"]
            return {"A": 0.2, "B": 0.9}

    result = scoring.score_packet(
        m.Packet(),
        [m.Candidate(label="A", content="one"), m.Candidate(label="B", content="two")],
        Judge(),
    )
    assert [s.score for s in result] == [0.2, 0.9]
    assert all(s.breakdown["judge_used"] == 1 for s in result)


def test_clear_rules_winner_avoids_judge():
    m, scoring, _, _, _ = modules()
    packet = m.Packet(directives=[m.Directive(polarity="do", text="concrete angle first")])

    class Judge:
        def judge(self, packet, candidates):
            pytest.fail("Clear rules winners must avoid paid judging")

    result = scoring.score_packet(
        packet,
        [
            m.Candidate(label="A", content="concrete angle first"),
            m.Candidate(label="B", content="unrelated filler"),
        ],
        Judge(),
    )
    assert result[0].score > result[1].score + 0.15


@pytest.mark.parametrize(
    "bad", [{"A": 0.9}, {"A": float("nan"), "B": 0.1}, {"A": 0.8, "B": 0.3, "injected": 1.0}]
)
def test_invalid_judge_results_visibly_degrade(bad):
    m, scoring, _, _, _ = modules()

    class Judge:
        def judge(self, packet, candidates):
            return bad

    result = scoring.score_packet(
        m.Packet(),
        [m.Candidate(label="A", content="a"), m.Candidate(label="B", content="b")],
        Judge(),
    )
    assert [s.score for s in result] == [0.5, 0.5]
    assert all(s.breakdown["judge_degraded"] == 1 for s in result)
    assert all(s.breakdown["judge_used"] == 0 for s in result)


def test_cloud_adapter_parses_fenced_json_without_exposing_secret():
    m, _, providers, config, _ = modules()
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url == "https://provider.invalid/v1/chat/completions"
        assert body["model"] == "configured-flash"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"scores":[{"ref":"A","score":0.8},{"ref":"B","score":0.2}]}\n```'
                        }
                    }
                ]
            },
        )

    settings = config.Settings(
        api_base="https://provider.invalid/v1",
        api_model="configured-flash",
        api_key="synthetic-test-token",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = providers.CloudProvider(settings, client=client)
        scores = provider.judge(
            m.Packet(), [m.Candidate(label="A", content="a"), m.Candidate(label="B", content="b")]
        )
    assert scores == {"A": 0.8, "B": 0.2}
    assert len(seen) == 1
    assert "synthetic-test-token" not in repr(settings)


def test_provider_failure_has_sanitized_error():
    m, _, providers, config, errors = modules()
    settings = config.Settings(api_base="https://provider.invalid/v1", api_model="configured-flash")
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(401, text="private body"))
        ) as client,
        pytest.raises(errors.DomainError) as exc,
    ):
        providers.CloudProvider(settings, client=client).judge(
            m.Packet(), [m.Candidate(label="A", content="x")]
        )
    assert exc.value.status_code == 503
    assert "private body" not in str(exc.value)


def test_exact_point_fifteen_gap_does_not_accidentally_judge_float_rounding():
    m, scoring, _, _, _ = modules()

    class Judge:
        def judge(self, packet, candidates):
            pytest.fail("An exact 0.15 gap must not trigger judging")

    rule = " ".join(f"term{i}" for i in range(20))
    result = scoring.score_packet(
        m.Packet(directives=[m.Directive(polarity="do", text=rule)]),
        [
            m.Candidate(label="A", content=" ".join(f"term{i}" for i in range(7))),
            m.Candidate(label="B", content=" ".join(f"term{i}" for i in range(4))),
        ],
        Judge(),
    )
    assert result[0].score == 0.35
    assert result[1].score == 0.2
    assert result[0].breakdown["judge_degraded"] == 0


def test_hard_constraint_ban_is_enforced_even_without_duplicate_directive():
    m, scoring, _, _, _ = modules()
    packet = m.Packet(constraints=[m.Constraint(hard=True, text='Never use "generic slop"')])
    result = scoring.score_packet(packet, [m.Candidate(label="A", content="generic slop")])
    assert result[0].score == 0.0
    assert result[0].violations


def test_numeric_word_limit_hard_constraint():
    m, scoring, _, _, _ = modules()
    packet = m.Packet(constraints=[m.Constraint(hard=True, text="At most 3 words")])
    result = scoring.score_packet(
        packet,
        [
            m.Candidate(label="A", content="one two three four"),
            m.Candidate(label="B", content="one two three"),
        ],
    )
    assert result[0].violations
    assert not result[1].violations


@pytest.mark.parametrize("texts", [["one", "two", "three"], ["one"], ["one", "", "three"]])
def test_cloud_generation_checks_complete_nonblank_batch(texts):
    m, _, providers, config, errors = modules()
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(json.loads(body["messages"][1]["content"])["count"])
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"candidates": texts})}}]}
        )

    settings = config.Settings(api_base="https://provider.invalid/v1", api_model="configured-flash")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = providers.CloudProvider(settings, client=client)
        if len(texts) == 3 and all(texts):
            assert provider.generate(m.Packet(), "draft", n=3) == ["one", "two", "three"]
        else:
            with pytest.raises(errors.DomainError):
                provider.generate(m.Packet(), "draft", n=3)
    assert seen == [3]


@pytest.mark.parametrize("directive", ['"generic slop"', "generic slop"])
def test_never_polarity_bans_without_an_english_prefix(directive):
    m, scoring, _, _, _ = modules()
    packet = m.Packet(
        directives=[m.Directive(polarity="never", text=directive)],
        constraints=[m.Constraint(hard=True, text=directive)],
    )
    result = scoring.score_packet(
        packet,
        [
            m.Candidate(label="A", content="generic slop"),
            m.Candidate(label="B", content="a clear specific angle"),
        ],
    )
    assert result[0].score == 0
    assert result[0].violations
    assert not result[1].violations


def test_quoted_rule_and_content_use_identical_unicode_normalization():
    m, scoring, _, _, _ = modules()
    packet = m.Packet(
        directives=[m.Directive(polarity="never", text='Never use "ＧＥＮＥＲＩＣ ＳＬＯＰ"')]
    )
    result = scoring.score_packet(packet, [m.Candidate(label="A", content="generic slop")])
    assert result[0].score == 0
    assert result[0].violations
