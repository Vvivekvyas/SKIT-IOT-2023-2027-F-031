"""
Thin backward-compatible wrapper around app.data.attack_taxonomy.

Week 5's Label Mapping + Attack Taxonomy deliverable moved the actual
taxonomy (category + severity + description per raw attack label) into
attack_taxonomy.py as the single source of truth. This file is kept so
existing imports (app.data.loaders, tests) keep working unchanged —
map_label() behaves exactly as before.
"""
from app.data.attack_taxonomy import taxonomy_for_dataset, get_category

CICIDS2017_LABEL_MAP: dict[str, str] = {
    e.raw_label: e.category.value for e in taxonomy_for_dataset("cicids2017")
}
CIC_IOT2023_LABEL_MAP: dict[str, str] = {
    e.raw_label: e.category.value for e in taxonomy_for_dataset("ciciot2023")
}


def map_label(raw_label: str, dataset: str) -> str:
    """
    dataset: "cicids2017" or "ciciot2023"
    Falls back to "Other" for anything not in the taxonomy so it's visible
    in class-balance reports rather than silently mis-binned.
    """
    return get_category(raw_label, dataset).value