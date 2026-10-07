"""Renders a list[SecurityFeatureRanking] as Markdown."""
from app.data.security_feature_analysis import SecurityFeatureRanking


def to_markdown(rankings: list[SecurityFeatureRanking]) -> str:
    lines = ["# Security-Oriented Feature Analysis", ""]

    lines.append(
        "Ranks each feature by how much the model leans on it (importance) "
        "combined with how easily an attacker could manipulate it "
        "(spoofability), and — where available — how often perturbing it "
        "flips the model's prediction, and how much its distribution shifts "
        "on held-out zero-day attacks."
    )
    lines.append("")

    lines.append("## Top Security Risk Features")
    lines.append("| Rank | Feature | Importance | Spoofability | Flip Rate % | Zero-Day Shift | Risk Score |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rankings[:20], start=1):
        flip = f"{r.flip_rate_pct}%" if r.flip_rate_pct is not None else "\u2014"
        shift = f"{r.distribution_shift}" if r.distribution_shift is not None else "\u2014"
        lines.append(f"| {i} | {r.feature} | {r.importance} | {r.spoofability} | {flip} | {shift} | {r.risk_score} |")
    lines.append("")

    high_risk = [
        r for r in rankings
        if r.spoofability == "attacker-controllable (per-packet field)" and r.importance > 0.3
    ]
    lines.append("## Recommendations")
    if high_risk:
        names = [r.feature for r in high_risk[:5]]
        lines.append(
            f"- {len(high_risk)} feature(s) are both model-important and "
            f"attacker-controllable: {names}. Consider pairing them with "
            f"aggregate/behavioral features so the model doesn't rely on "
            f"easily-faked per-packet fields alone."
        )
    high_flip = [r for r in rankings if r.flip_rate_pct is not None and r.flip_rate_pct > 20]
    if high_flip:
        names = [r.feature for r in high_flip[:5]]
        lines.append(
            f"- {len(high_flip)} feature(s) flip the prediction for >20% of "
            f"samples under a small perturbation: {names}. These are "
            f"practical evasion vectors — worth adversarial-training or "
            f"input-validation attention."
        )
    high_shift = [r for r in rankings if r.distribution_shift is not None and r.distribution_shift > 1.0]
    if high_shift:
        names = [r.feature for r in high_shift[:5]]
        lines.append(
            f"- {len(high_shift)} feature(s) shift by more than 1 std on "
            f"held-out zero-day attacks: {names}. Importance learned from "
            f"known attacks may not transfer well for these on genuinely "
            f"new attack types."
        )
    if not high_risk and not high_flip and not high_shift:
        lines.append("- No major security concerns flagged in this pass.")

    return "\n".join(lines)