# VARIANT: taste-3 — temporal graph

**Thesis under test:** does taste need *relations*? Standards as a typed
graph — decisions, rules, and exemplars as nodes with supersession edges and
validity windows (Graphiti/Zep-shaped). Retrieval traverses, not just ranks.

## Permutation

| Axis | Choice | Divergence from control |
|---|---|---|
| A schema | **A3 temporal knowledge graph** | the axis under test |
| B retrieval | B4 graph traversal (related standards via edges) | relational recall |
| C embedding | C1 MiniLM | held constant w/ control |
| D scorer | D4 hybrid (rules gate → judge on borderline) | judge as tiebreaker |
| E interview | E1 adaptive | owner amendment: question unresolved gaps under Contract v1 |
| F hone | F1 BoN-3 | held w/ control |
| G storage | G1 SQLite WAL (edge table + node table) | held w/ control |
| H memory | H2 temporal supersession (validity windows, drift) | taste ages explicitly |
| I injection | I3 contrastive pairs (rejected vs chosen + why) | strongest teaching signal |
| J pipeline | J1 fixed segments | held w/ control |

## The question it answers

**Owner amendment — 2026-10-12:** The original E3/no-interview setting
conflicted with locked Contract v1. The owner chose v1 as authoritative,
so this variant now uses E1/adaptive interview. Cohort metadata and
comparisons must disclose the changed E axis.

Taste drifts — "we used to do X, now Y." If graph supersession + validity
windows handle drift better than flat replacement, A3 proves the added
schema complexity. If not, flat wins on simplicity.

## Implemented boundaries

C1 is actual 384-dimensional CPU ONNX MiniLM with a repository-local model
cache. B4 includes its top-five seeds, takes one outgoing eligible hop, uses
maximum path scores and stable ID ties, and retains twelve relevance results
plus every effective never-rule. H2 requires local exact-revision approval;
source declarations alone never activate imported standards.

D4 rules are a disclosed lexical/phrase/word-limit heuristic. An optional
operator-configured cloud judge handles eligible borderline cases only;
unavailable judging is visibly degraded. F1 emits three candidates under a
durable fifteen-attempt task budget. Its default P3 generator is mechanical
`baseline-p3`, not a live generation model. No live provider quality or cohort
evaluation is claimed. E1 uses deterministic unresolved dimensions and explicit
conflicts; I3 includes only complete, immutable chosen/rejected content pairs.
