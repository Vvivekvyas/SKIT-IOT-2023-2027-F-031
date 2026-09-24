"""Renders a FeatureQualityReport as Markdown for the team/supervisor to read."""
from app.data.feature_quality import FeatureQualityReport


def to_markdown(report: FeatureQualityReport) -> str:
    lines = ["# Feature Quality & Security Analysis Report", ""]

    lines.append("## Dataset Overview")
    lines.append(f"- Rows: {report.n_rows}")
    lines.append(f"- Features analyzed: {report.n_features}")
    lines.append(f"- Exact duplicate rows remaining: {report.exact_duplicate_rows}")
    lines.append(f"- Class balance: {report.class_balance}")
    lines.append("")

    lines.append("## Recommendations")
    for r in report.recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Security / Leakage Risk")
    lines.append(f"- Identity-like columns in feature set: {report.identity_leak_columns or 'None'}")
    if report.single_feature_leak_risk:
        lines.append("- Features with suspiciously high single-feature separation (AUC):")
        for name, auc in report.single_feature_leak_risk:
            lines.append(f"  - `{name}`: AUC={auc}")
    else:
        lines.append("- No single feature achieves near-perfect class separation alone.")
    lines.append("")

    lines.append("## Feature Importance (Random Forest)")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for name, score in report.importance_ranking:
        lines.append(f"| {name} | {score} |")
    lines.append("")

    lines.append("## Feature Relevance (Mutual Information)")
    lines.append("| Feature | Mutual Info |")
    lines.append("|---|---|")
    for name, score in report.mutual_info_ranking:
        lines.append(f"| {name} | {score} |")
    lines.append("")

    lines.append("## Redundant Feature Pairs (|correlation| >= 0.95)")
    if report.redundant_pairs:
        lines.append("| Feature A | Feature B | Correlation |")
        lines.append("|---|---|---|")
        for a, b, corr in report.redundant_pairs:
            lines.append(f"| {a} | {b} | {corr} |")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Near-Zero-Variance Features")
    lines.append(", ".join(report.near_zero_variance) if report.near_zero_variance else "None found.")
    lines.append("")

    lines.append("## Outlier Proportion per Feature (IQR rule)")
    lines.append("| Feature | % Outliers |")
    lines.append("|---|---|")
    for name, pct in sorted(report.outlier_pct.items(), key=lambda x: -x[1]):
        lines.append(f"| {name} | {pct}% |")
    lines.append("")

    lines.append("## Missing Values (post-cleaning — should be ~0)")
    lines.append("| Feature | % Missing |")
    lines.append("|---|---|")
    for name, pct in report.missing_pct.items():
        lines.append(f"| {name} | {pct}% |")

    return "\n".join(lines)