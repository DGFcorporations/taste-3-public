"""Synthetic task coverage; approved taste must actually resolve a dimension."""

from datetime import UTC, datetime
from importlib import import_module, util

from taste.core import StandardsEngine
from taste.models import InterviewRequest, Standard
from taste.store import Store

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class FixedEmbedder:
    model_name = "test-only:fixed-vectors"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def module(name):
    assert util.find_spec(name) is not None, f"Required task-learning module {name} is missing"
    return import_module(name)


def engine_for(store):
    return StandardsEngine(store, FixedEmbedder())


def approved_rule(store, id="s", **changes):
    data = dict(id=id, profile_id="owner", category="structure", polarity="do",
                directive="Use clear headings", rationale="Synthetic clarity", created=NOW, updated=NOW)
    data.update(changes)
    item = store.propose(Standard.model_validate(data))
    return store.approve(item.profile_id, item.id, item.rationale, owner_confirmed=True,
                         expected_updated=item.updated)


def interview(store, ctx):
    return module("taste.interview").interview(engine_for(store), InterviewRequest(task_ctx=ctx))


def test_fresh_typed_task_gets_three_substantive_unresolved_questions():
    with Store(clock=lambda: NOW) as store:
        response = interview(store, {"type": "write_doc"})
        assert len(response.questions) == 3
        assert response.coverage["resolved"] == []
        assert set(response.coverage["gaps"]) == {"goal", "audience", "output_shape", "success_criteria", "constraints"}
        assert [q.resolves for q in response.questions] == ["goal", "audience", "output_shape"]
        assert all(len(q.text) > 15 and q.id for q in response.questions)


def test_explicit_context_closes_only_supplied_nonempty_dimensions():
    with Store(clock=lambda: NOW) as store:
        response = interview(store, {"goal": "Explain", "audience": "Operators", "output_shape": "Memo",
                                     "success_criteria": "", "constraints": []})
        assert response.coverage == {"resolved": ["goal", "audience", "output_shape"],
                                     "gaps": ["success_criteria", "constraints"]}
        assert [q.resolves for q in response.questions] == ["success_criteria", "constraints"]


def test_all_relevant_dimensions_resolved_yields_zero_questions():
    context = {"goal": "Explain", "audience": "Operators", "output_shape": "Memo",
               "success_criteria": ["Actionable"], "constraints": ["Under 200 words"]}
    with Store(clock=lambda: NOW) as store:
        response = interview(store, context)
        assert response.questions == []
        assert response.coverage["gaps"] == []


def test_category_alone_does_not_claim_format_or_goal_is_resolved():
    with Store(clock=lambda: NOW) as store:
        approved_rule(store)
        response = interview(store, {})
        assert "goal" in response.coverage["gaps"]
        assert "output_shape" in response.coverage["gaps"]


def test_explicit_approved_coverage_resolves_gaps_but_proposal_does_not():
    with Store(clock=lambda: NOW) as store:
        item = store.propose(Standard(category="aesthetic", polarity="do", directive="Use calm tone",
                                      rationale="Synthetic", variant_meta={"coverage": ["audience", "output_shape"]}))
        assert "audience" in interview(store, {}).coverage["gaps"]
        store.approve("owner", item.id, "Synthetic", owner_confirmed=True, expected_updated=item.updated)
        response = interview(store, {})
        assert "audience" in response.coverage["resolved"]
        assert "output_shape" in response.coverage["resolved"]
        assert "goal" in response.coverage["gaps"]


def test_opposed_approved_rules_leave_a_visible_conflict_question():
    with Store(clock=lambda: NOW) as store:
        approved_rule(store, "allow", directive="Use headings")
        approved_rule(store, "ban", polarity="never", directive="Never use headings")
        response = interview(store, {})
        assert any("conflict" in gap for gap in response.coverage["gaps"])
        assert any("conflict" in q.resolves for q in response.questions)
        assert interview(store, {}).model_dump() == response.model_dump()


def test_explicit_graph_contradiction_is_visible_even_when_prose_differs():
    with Store(clock=lambda: NOW) as store:
        approved_rule(store, "a", directive="Use short sentences")
        approved_rule(store, "b", directive="Use detailed explanations")
        store.add_edge("owner", "a", "b", "contradicts")
        assert any("conflict" in gap for gap in interview(store, {}).coverage["gaps"])


def test_explicitly_inapplicable_dimensions_are_not_invented_gaps():
    with Store() as store:
        response = interview(store, {"goal": "Run a private self-check", "not_applicable":
            ["audience", "output_shape", "success_criteria", "constraints"]})
        assert response.coverage == {"resolved": ["goal"], "gaps": []}
        assert response.questions == []
