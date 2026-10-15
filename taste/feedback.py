"""One atomic persistence call records rationale and real content evidence."""

from __future__ import annotations

from pydantic import ValidationError

from taste.errors import DomainError
from taste.models import FeedbackRequest, FeedbackResponse
from taste.store import Store


def feedback(store: Store, req: FeedbackRequest) -> FeedbackResponse:
    try:
        checked = FeedbackRequest.model_validate(req.model_dump(mode="json"))
    except ValidationError:
        raise DomainError("invalid_rationale", "Feedback requires valid context and nonblank rationale.") from None
    record, proposals = store.record_feedback(checked.profile_id, checked.task_ctx, checked.pick,
                                               checked.rationale, checked.rejected)
    return FeedbackResponse(accepted=True, record_id=record.id, extracted={"candidates": len(proposals)})
