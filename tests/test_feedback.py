"""Conservative extraction and feedback atomicity on synthetic stores."""

import pytest

from taste.errors import DomainError
from taste.models import Candidate, FeedbackRequest, IngestRequest
from taste.store import Store
from tests.test_interview import NOW, module


def test_ingest_extracts_explicit_rules_with_provisional_restatements():
    with Store(clock=lambda: NOW) as store:
        request = IngestRequest(docs=[{"id": "synthetic", "format": "md", "text":
            '# Synthetic\n- Never use "generic opener".\n- Lead with results because readers scan.\n- Prefer short sentences.\nRevenue grew yesterday.'}])
        response = module("taste.ingest").ingest(store, request)
        assert len(response.candidates) == len(response.restatements) == 3
        assert all(c.state == "provisional" for c in response.candidates)
        assert {c.polarity for c in response.candidates} == {"do", "never"}
        assert all(r.implications and r.layer.id in {c.id for c in response.candidates} for r in response.restatements)
        assert all(not row["_local_approved"] for row in store.snapshot("owner").standards)
        reasoned = next(c for c in response.candidates if "Lead" in c.directive)
        assert reasoned.rationale == "readers scan."


def test_ingest_plain_prose_does_not_turn_facts_into_rules_or_parse_markup():
    with Store(clock=lambda: NOW) as store:
        response = module("taste.ingest").ingest(store, IngestRequest(docs=[
            {"id": "facts", "format": "txt", "text": "Sales rose. Use clear headings. The team met."},
            {"id": "markup", "format": "doc", "text": '<script>Use secrets</script>\n!!python/object/apply:os.system [echo synthetic]'},
        ]))
        assert [c.directive for c in response.candidates] == ["Use clear headings."]


def test_ingest_deduplicates_repeated_rules_without_false_activation():
    with Store(clock=lambda: NOW) as store:
        request = IngestRequest(docs=[{"id": "one", "text": "Use clear headings.\nUse clear headings."}])
        response = module("taste.ingest").ingest(store, request)
        assert len(response.candidates) == 1
        again = module("taste.ingest").ingest(store, request)
        assert again.candidates == []
        assert len(store.list_standards()) == 1


@pytest.mark.parametrize("bad", [{"id": "bad", "text": "Use plain text", "format": "html"},
                                {"id": "bad", "format": "md"}, {"text": "Use plain text"}])
def test_ingest_validates_all_documents_before_writing_any_proposal(bad):
    with Store(clock=lambda: NOW) as store:
        with pytest.raises(DomainError) as caught:
            module("taste.ingest").ingest(store, IngestRequest(docs=[{"id": "good", "text": "Use clear headings"}, bad]))
        assert caught.value.status_code == 422
        assert store.list_standards() == []


def test_feedback_records_known_content_and_provisional_extraction_in_one_transaction():
    with Store(clock=lambda: NOW) as store:
        store.register_candidates("owner", "t", [Candidate(label="A", content="Chosen text"), Candidate(label="B", content="Rejected text")])
        response = module("taste.feedback").feedback(store, FeedbackRequest(task_ctx={"task_ref": "t"}, pick="A", rationale="Prefer clear results", rejected=["B"]))
        assert response.accepted is True and response.record_id
        assert response.extracted == {"candidates": 1}
        snapshot = store.snapshot("owner")
        assert len(snapshot.standards) == 1 and snapshot.standards[0]["state"] == "provisional"
        assert snapshot.exemplars[0]["decision_id"] == response.record_id
        assert snapshot.exemplars[0]["text"] == "Chosen text"
        assert snapshot.exemplars[0]["rejected"] == ["Rejected text"]


def test_pick_only_feedback_does_not_fabricate_content_from_label():
    with Store(clock=lambda: NOW) as store:
        response = module("taste.feedback").feedback(store, FeedbackRequest(pick="A", rationale="Prefer clarity", rejected=["Literal rejected"] ))
        assert response.accepted is True
        assert store.snapshot("owner").exemplars[0]["text"] is None


def test_constructed_invalid_feedback_cannot_bypass_rationale_or_create_rows():
    with Store() as store:
        invalid = FeedbackRequest.model_construct(profile_id="owner", task_ctx={}, pick="A", rationale=" \n", rejected=[])
        with pytest.raises(DomainError) as caught:
            module("taste.feedback").feedback(store, invalid)
        assert caught.value.status_code == 422
        assert store.snapshot("owner").revision == 0
        assert store.snapshot("owner").exemplars == []


def test_markdown_code_examples_are_not_mistaken_for_owner_rules():
    with Store() as store:
        response = module("taste.ingest").ingest(store, IngestRequest(docs=[{"id": "code", "format": "md", "text":
            '```text\nNever use synthetic example\n```\n    Never use synthetic code\nUse clear headings.'}]))
        assert [candidate.directive for candidate in response.candidates] == ["Use clear headings."]
