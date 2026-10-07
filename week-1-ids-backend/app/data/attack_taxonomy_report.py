"""Renders the attack taxonomy (and optional coverage-gap findings) as Markdown."""
from app.data.attack_taxonomy import (
    CATEGORY_DESCRIPTIONS,
    AttackCategory,
    taxonomy_for_dataset,
)


def _taxonomy_table(dataset: str) -> list[str]:
    lines = [f"### {dataset}", "", "| Raw Label | Category | Severity | Description |", "|---|---|---|---|"]
    for entry in sorted(taxonomy_for_dataset(dataset), key=lambda e: (e.category.value, e.raw_label)):
        lines.append(f"| {entry.raw_label} | {entry.category.value} | {entry.severity.value} | {entry.description} |")
    lines.append("")
    return lines


def to_markdown(coverage_gaps: dict[str, list[str]] | None = None) -> str:
    """
    coverage_gaps: optional {"cicids2017": [...], "ciciot2023": [...]} from
    attack_taxonomy.check_coverage(), to include an "unmapped labels found
    in data" section. Pass None to render just the taxonomy reference.
    """
    lines = ["# Attack Taxonomy", ""]

    lines.append("## Category Definitions")
    for category in AttackCategory:
        lines.append(f"- **{category.value}** — {CATEGORY_DESCRIPTIONS[category]}")
    lines.append("")

    if coverage_gaps:
        lines.append("## Coverage Check (labels found in data but NOT in this taxonomy)")
        any_gaps = False
        for dataset, gaps in coverage_gaps.items():
            if gaps:
                any_gaps = True
                lines.append(f"- **{dataset}**: {gaps} — these currently fall back to category 'Other'. Add them to `app/data/attack_taxonomy.py`.")
        if not any_gaps:
            lines.append("No gaps found — every label seen in the data is covered by the taxonomy.")
        lines.append("")

    lines.append("## Full Taxonomy Reference")
    lines.append("")
    lines.extend(_taxonomy_table("cicids2017"))
    lines.extend(_taxonomy_table("ciciot2023"))

    return "\n".join(lines)