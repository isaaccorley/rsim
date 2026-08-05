from __future__ import annotations

# Default matcher roster for ``spacenet9_matcher_benchmark`` and
# ``srif_matcher_benchmark`` when ``--matchers`` is omitted: a curated subset
# of the matchers reported in the paper. Per-table scripts under ``scripts/``
# pass their own explicit ``--matchers`` list when they need a different set.
DEFAULT_MATCHERS: list[str] = [
    "duster",
    "romav2",
    "xfeat",
    "superpoint-lightglue",
    "minima-roma-tiny",
    "tiny-roma",
    "loftr",
    "master",
    "roma+loftr",
    "roma",
    "roma+tiny-roma",
    "minima-roma",
]

# Classical / Kornia-backed baselines used for the SARptical retrieval table.
SARPTICAL_BASELINE_MATCHERS: list[str] = [
    "doghardnet-lightglue",
    "sift-nn",
    "disk-lightglue",
]
