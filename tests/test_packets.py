"""Stable, bounded packets follow approved links and real frozen content."""

from taste.models import Candidate, PacketRequest, Standard
from taste.store import Store
from tests.test_interview import NOW, approved_rule, engine_for, module


def packet(store, ctx=None, **kwargs):
    return module("taste.inject").compile_packet(engine_for(store), PacketRequest(task_ctx=ctx or {}, **kwargs))


def test_empty_packet_has_fixed_segment_freedoms_and_stable_revision_metadata():
    with Store(clock=lambda: NOW) as store:
        response = packet(store, {"type": "write_doc"})
        assert response.variant_meta["segments"] == ["lead", "structure", "detail", "close"]
        assert len(response.free_degrees) == 4
        assert all(any(f.startswith(segment + ":") and "constraint" in f for f in response.free_degrees)
                   for segment in response.variant_meta["segments"])
        assert response.variant_meta["corpus_revision"] == 0
        assert response.variant_meta["evaluation_time"] == "1970-01-01T00:00:00+00:00"
        assert response.variant_meta["model"] == "test-only:fixed-vectors"
        assert packet(store, {"type": "write_doc"}).model_dump(mode="json") == response.model_dump(mode="json")


def test_explicit_segment_retains_only_its_bounded_freedom():
    with Store() as store:
        response = packet(store, {"type": "write_doc", "segment": "detail"})
        assert response.variant_meta["segments"] == ["detail"]
        assert len(response.free_degrees) == 1 and response.free_degrees[0].startswith("detail:")


def test_never_rules_are_hard_deduplicated_and_guidelines_cannot_override_them():
    with Store(clock=lambda: NOW) as store:
        approved_rule(store, "ban", polarity="never", directive='Never use "generic opener"', variant_meta={"private_note": "internal-sentinel"})
        approved_rule(store, "good", directive="Lead with results")
        response = packet(store, guidelines=['Use "generic opener"'])
        assert [d.standard_id for d in response.directives].count("ban") == 1
        assert any(c.hard and c.text == 'Never use "generic opener"' for c in response.constraints)
        assert any(not c.hard and c.text == 'Use "generic opener"' for c in response.constraints)
        assert response.variant_meta["standard_ids"] == ["ban", "good"]
        dumped = str(response.model_dump())
        assert "_local_approved" not in dumped and "internal-sentinel" not in dumped and "aliases" not in dumped


def test_foreign_unapproved_and_superseded_rows_never_become_packet_directives():
    with Store(clock=lambda: NOW) as store:
        old = approved_rule(store, "old", directive="Use old format")
        new = store.propose(Standard(id="new", category="structure", polarity="do", directive="Use new format", rationale="Synthetic"))
        store.approve("owner", "new", "Synthetic", owner_confirmed=True, expected_updated=new.updated, supersedes=[old.id], expected_targets={old.id: old.updated})
        store.propose(Standard(id="unapproved", category="structure", polarity="never", directive="Never share text"))
        approved_rule(store, "foreign", profile_id="other", directive="Use foreign format")
        response = packet(store)
        assert [d.standard_id for d in response.directives] == ["new"]


def test_complete_contrastive_pair_requires_linked_locally_approved_taste():
    with Store(clock=lambda: NOW) as store:
        store.register_candidates("owner", "t", [Candidate(label="A", content="Real chosen"), Candidate(label="B", content="Real rejected")])
        record, proposals = store.record_feedback("owner", {"task_ref": "t"}, "A", "Clarity wins", ["B"])
        assert packet(store).exemplars == []
        proposal = proposals[0]
        store.approve("owner", proposal.id, "Clarity wins", owner_confirmed=True, expected_updated=proposal.updated)
        examples = packet(store).exemplars
        assert len(examples) == 1
        assert examples[0].text == "Real chosen" and examples[0].why == "Clarity wins"
        assert examples[0].rejected == "Real rejected"
        assert examples[0].text != "A" and record.id


def test_unresolved_or_single_sided_examples_never_fabricate_a_pair():
    with Store(clock=lambda: NOW) as store:
        _, proposals = store.record_feedback("owner", {"task_ref": "t"}, "Missing", "Clear structure wins", ["Literal reject"])
        store.approve("owner", proposals[0].id, "Clear structure wins", owner_confirmed=True, expected_updated=proposals[0].updated)
        assert packet(store).exemplars == []
        store.register_candidates("owner", "t", [Candidate(label="Missing", content="Late chosen")])
        assert packet(store).exemplars == []


def test_packet_exemplars_use_same_snapshot_without_later_reads(monkeypatch):
    with Store(clock=lambda: NOW) as store:
        store.register_candidates("owner", "t", [Candidate(label="A", content="Chosen")])
        _, proposals = store.record_feedback("owner", {"task_ref": "t"}, "A", "Prefer clarity", ["Rejected"])
        store.approve("owner", proposals[0].id, "Prefer clarity", owner_confirmed=True, expected_updated=proposals[0].updated)
        policy = engine_for(store)
        context = policy.context("owner", {})
        def coherent(*args):
            return context
        def forbidden(*args):
            raise AssertionError("Packet attempted a second store read")
        monkeypatch.setattr(policy, "context", coherent)
        monkeypatch.setattr(store, "snapshot", forbidden)
        monkeypatch.setattr(store, "list_standards", forbidden)
        result = module("taste.inject").compile_packet(policy, PacketRequest())
        assert result.exemplars[0].text == "Chosen"


def test_packet_conflicting_approved_requirements_are_disclosed():
    with Store(clock=lambda: NOW) as store:
        approved_rule(store, "do", directive="Use headings")
        approved_rule(store, "never", polarity="never", directive="Never use headings")
        assert any("conflict" in gap for gap in packet(store).variant_meta["unresolved_gaps"])


def test_packet_question_answers_close_coverage_without_changing_wire_context():
    context = {"goal": "", "type": "write_doc"}
    answers = {"q_goal": "Explain the result", "q_audience": "Operators", "q_output_shape": "Memo",
               "q_success_criteria": "Actionable", "q_constraints": "Under 200 words"}
    with Store() as store:
        response = packet(store, context, answers=answers)
        assert response.task_ctx == context
        assert response.variant_meta["unresolved_gaps"] == []
