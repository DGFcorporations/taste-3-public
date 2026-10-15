"""Compile one coherent taste snapshot into a stable v1 packet."""

from __future__ import annotations

from taste.core import StandardsEngine
from taste.interview import applied_rows, clarified_context, coverage_for
from taste.models import Constraint, Directive, Exemplar, Packet, PacketRequest
from taste.segments import segments_for

AXES = {"A": "A3", "B": "B4", "C": "C1", "D": "D4", "E": "E1",
        "F": "F1", "G": "G1", "H": "H2", "I": "I3", "J": "J1"}


def compile_packet(engine: StandardsEngine, req: PacketRequest) -> Packet:
    query = clarified_context(req.task_ctx, req.answers)
    if req.answers:
        query["answers"] = req.answers
    if req.guidelines:
        query["guidelines"] = req.guidelines
    context = engine.context(req.profile_id, query)
    rows = applied_rows(context)
    ids = {row["id"] for row in rows}
    directives = [Directive(polarity=row["polarity"], text=row["directive"], standard_id=row["id"]) for row in rows]
    constraints = [Constraint(hard=True, text=row["directive"]) for row in rows if row["polarity"] == "never"]
    constraints.extend(Constraint(hard=False, text=guideline) for guideline in req.guidelines)
    linked = {edge["src"] for edge in context.snapshot.edges
              if edge.get("profile_id") == req.profile_id
              and edge.get("edge_type", edge.get("type")) == "exemplar_of" and edge["dst"] in ids}
    exemplars = []
    for example in sorted(context.snapshot.exemplars, key=lambda entry: entry["id"]):
        if example["id"] not in linked:
            continue
        text, why = example.get("text"), example.get("why")
        if not isinstance(text, str) or not text.strip() or not isinstance(why, str) or not why.strip():
            continue
        for rejected in example.get("rejected", []):
            if isinstance(rejected, str) and rejected.strip():
                exemplars.append(Exemplar(text=text, why=why, rejected=rejected))
    explicit = req.task_ctx.get("segment")
    task_type = req.task_ctx.get("type")
    segments = [explicit] if isinstance(explicit, str) and explicit.strip() else segments_for(task_type if isinstance(task_type, str) else None)
    freedoms = [f"{segment}: choose phrasing and examples within the approved hard constraints." for segment in segments]
    clarified = clarified_context(req.task_ctx, req.answers)
    return Packet(profile_id=req.profile_id, task_ctx=req.task_ctx, directives=directives,
                  exemplars=exemplars, constraints=constraints, free_degrees=freedoms,
                  variant_meta={"axes": AXES, "corpus_revision": context.snapshot.revision,
                                "evaluation_time": context.as_of.isoformat(), "model": context.model_name,
                                "standard_ids": sorted(ids), "segments": segments,
                                "unresolved_gaps": coverage_for(context, clarified)["gaps"]})
