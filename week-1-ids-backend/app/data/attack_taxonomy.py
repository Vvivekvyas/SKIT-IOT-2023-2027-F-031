"""
Formal attack taxonomy for the IDS project — the single source of truth for
"what attack is this, which of the 5 output categories does it belong to,
and how severe is it".

Why this exists on top of Week 2's label_mapping.py: that file only mapped
raw label -> category (a flat dict). This module adds:
  - a severity rating per attack type (feeds the dashboard's alert panel —
    currently every alert is hardcoded "high" in predict.py; this gives the
    real data to fix that)
  - a one-line description per category (for the project report / viva)
  - a coverage-check function so the team can verify every label actually
    present in the raw CSVs is accounted for, instead of silently falling
    back to "Other"

label_mapping.py now delegates to this module (see that file) so nothing
else needs to change.
"""
from dataclasses import dataclass
from enum import Enum


class AttackCategory(str, Enum):
    NORMAL = "Normal"
    DOS = "DoS"
    PROBE = "Probe"
    R2L = "R2L"
    U2R = "U2R"
    OTHER = "Other"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


CATEGORY_DESCRIPTIONS: dict[AttackCategory, str] = {
    AttackCategory.NORMAL: "Benign traffic — no attack.",
    AttackCategory.DOS: (
        "Denial-of-Service / Distributed DoS: flooding or resource-exhaustion "
        "attacks that degrade or deny service availability."
    ),
    AttackCategory.PROBE: (
        "Reconnaissance/scanning: an attacker mapping hosts, ports, or "
        "vulnerabilities before a targeted attack."
    ),
    AttackCategory.R2L: (
        "Remote-to-Local: an outside attacker gaining unauthorized access or "
        "executing commands on a host over the network (brute force, "
        "injection, spoofing)."
    ),
    AttackCategory.U2R: (
        "User-to-Root / host compromise: malware or botnet activity that has "
        "already gained a foothold and is escalating privileges or acting as "
        "a compromised host."
    ),
    AttackCategory.OTHER: "Not yet mapped to a category — needs manual triage.",
}

# Fallback severity by category, used only when a specific label isn't in
# the taxonomy table below (see check_coverage()).
_DEFAULT_SEVERITY_BY_CATEGORY: dict[AttackCategory, Severity] = {
    AttackCategory.NORMAL: Severity.NONE,
    AttackCategory.DOS: Severity.HIGH,
    AttackCategory.PROBE: Severity.LOW,
    AttackCategory.R2L: Severity.MEDIUM,
    AttackCategory.U2R: Severity.CRITICAL,
    AttackCategory.OTHER: Severity.MEDIUM,  # unknown — treat cautiously, not as "none"
}


@dataclass(frozen=True)
class AttackTypeInfo:
    raw_label: str
    dataset: str  # "cicids2017" | "ciciot2023"
    category: AttackCategory
    severity: Severity
    description: str


def _e(raw_label: str, dataset: str, category: AttackCategory, severity: Severity, description: str) -> AttackTypeInfo:
    return AttackTypeInfo(raw_label, dataset, category, severity, description)


# --- Full taxonomy table -----------------------------------------------
# Severity is judged per attack type, not just inherited from the category —
# e.g. Heartbleed sits in the DoS bucket for the classic 5-category scheme
# but is actually a memory-disclosure vulnerability, so it's rated critical
# rather than the DoS-default "high".
_TAXONOMY: list[AttackTypeInfo] = [
    # ---------------- CICIDS2017 ----------------
    _e("BENIGN", "cicids2017", AttackCategory.NORMAL, Severity.NONE, "Benign/background traffic."),
    _e("DoS Hulk", "cicids2017", AttackCategory.DOS, Severity.HIGH, "HTTP flood DoS tool (Hulk)."),
    _e("DoS GoldenEye", "cicids2017", AttackCategory.DOS, Severity.HIGH, "HTTP flood DoS tool (GoldenEye)."),
    _e("DoS slowloris", "cicids2017", AttackCategory.DOS, Severity.MEDIUM, "Low-bandwidth slow-connection DoS, harder to spot by volume alone."),
    _e("DoS Slowhttptest", "cicids2017", AttackCategory.DOS, Severity.MEDIUM, "Low-bandwidth slow-HTTP DoS."),
    _e("DDoS", "cicids2017", AttackCategory.DOS, Severity.CRITICAL, "Distributed DoS — multiple sources, larger blast radius than single-host DoS."),
    _e("Heartbleed", "cicids2017", AttackCategory.DOS, Severity.CRITICAL, "OpenSSL heartbeat memory-disclosure vulnerability — can leak private keys/credentials, not just availability impact despite the DoS bucket."),
    _e("PortScan", "cicids2017", AttackCategory.PROBE, Severity.LOW, "Port scanning — reconnaissance, no direct damage but precedes targeted attacks."),
    _e("FTP-Patator", "cicids2017", AttackCategory.R2L, Severity.MEDIUM, "FTP credential brute-force."),
    _e("SSH-Patator", "cicids2017", AttackCategory.R2L, Severity.MEDIUM, "SSH credential brute-force."),
    _e("Web Attack \u2013 Brute Force", "cicids2017", AttackCategory.R2L, Severity.MEDIUM, "Web login brute-force."),
    _e("Web Attack \u2013 XSS", "cicids2017", AttackCategory.R2L, Severity.HIGH, "Cross-site scripting injection."),
    _e("Web Attack \u2013 Sql Injection", "cicids2017", AttackCategory.R2L, Severity.HIGH, "SQL injection — can lead to full data exposure."),
    _e("Infiltration", "cicids2017", AttackCategory.U2R, Severity.CRITICAL, "Attacker has already infiltrated a host via a dropped payload."),
    _e("Bot", "cicids2017", AttackCategory.U2R, Severity.HIGH, "Host compromised and acting as part of a botnet."),

    # ---------------- CIC-IoT2023 ----------------
    _e("BenignTraffic", "ciciot2023", AttackCategory.NORMAL, Severity.NONE, "Benign/background IoT traffic."),

    _e("DDoS-RSTFINFlood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "TCP RST/FIN flood DDoS."),
    _e("DDoS-PSHACK_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "TCP PSH-ACK flood DDoS."),
    _e("DDoS-SYN_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "TCP SYN flood DDoS."),
    _e("DDoS-UDP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "UDP flood DDoS."),
    _e("DDoS-TCP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "Generic TCP flood DDoS."),
    _e("DDoS-ICMP_Flood", "ciciot2023", AttackCategory.DOS, Severity.MEDIUM, "ICMP flood DDoS."),
    _e("DDoS-SynonymousIP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "DDoS using spoofed/synonymous source IPs."),
    _e("DDoS-ACK_Fragmentation", "ciciot2023", AttackCategory.DOS, Severity.MEDIUM, "Fragmented ACK packet flood."),
    _e("DDoS-UDP_Fragmentation", "ciciot2023", AttackCategory.DOS, Severity.MEDIUM, "Fragmented UDP packet flood."),
    _e("DDoS-ICMP_Fragmentation", "ciciot2023", AttackCategory.DOS, Severity.MEDIUM, "Fragmented ICMP packet flood."),
    _e("DDoS-SlowLoris", "ciciot2023", AttackCategory.DOS, Severity.MEDIUM, "Low-bandwidth slow-connection DDoS variant."),
    _e("DDoS-HTTP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "HTTP flood DDoS."),
    _e("DoS-UDP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "Single-source UDP flood DoS."),
    _e("DoS-SYN_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "Single-source TCP SYN flood DoS."),
    _e("DoS-TCP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "Single-source TCP flood DoS."),
    _e("DoS-HTTP_Flood", "ciciot2023", AttackCategory.DOS, Severity.HIGH, "Single-source HTTP flood DoS."),

    _e("Recon-PingSweep", "ciciot2023", AttackCategory.PROBE, Severity.LOW, "ICMP ping sweep — host discovery."),
    _e("Recon-OSScan", "ciciot2023", AttackCategory.PROBE, Severity.LOW, "OS fingerprinting scan."),
    _e("Recon-PortScan", "ciciot2023", AttackCategory.PROBE, Severity.LOW, "Port scanning."),
    _e("VulnerabilityScan", "ciciot2023", AttackCategory.PROBE, Severity.MEDIUM, "Active vulnerability scanning — more targeted than a plain port scan."),
    _e("Recon-HostDiscovery", "ciciot2023", AttackCategory.PROBE, Severity.LOW, "Host discovery sweep."),

    _e("DNS_Spoofing", "ciciot2023", AttackCategory.R2L, Severity.HIGH, "DNS response spoofing — can redirect victims to malicious hosts."),
    _e("MITM-ArpSpoofing", "ciciot2023", AttackCategory.R2L, Severity.HIGH, "ARP spoofing for man-in-the-middle positioning."),
    _e("BrowserHijacking", "ciciot2023", AttackCategory.R2L, Severity.HIGH, "Browser session/settings hijacking."),
    _e("Backdoor_Malware", "ciciot2023", AttackCategory.R2L, Severity.CRITICAL, "Backdoor malware deployment — persistent unauthorized access."),
    _e("CommandInjection", "ciciot2023", AttackCategory.R2L, Severity.CRITICAL, "OS command injection — can lead to full host compromise."),
    _e("SqlInjection", "ciciot2023", AttackCategory.R2L, Severity.HIGH, "SQL injection."),
    _e("XSS", "ciciot2023", AttackCategory.R2L, Severity.HIGH, "Cross-site scripting injection."),
    _e("Uploading_Attack", "ciciot2023", AttackCategory.R2L, Severity.HIGH, "Malicious file upload."),
    _e("DictionaryBruteForce", "ciciot2023", AttackCategory.R2L, Severity.MEDIUM, "Dictionary-based credential brute-force."),

    _e("Mirai-greeth_flood", "ciciot2023", AttackCategory.U2R, Severity.CRITICAL, "Mirai botnet GRE-over-Ethernet flood — host already compromised and attacking."),
    _e("Mirai-greip_flood", "ciciot2023", AttackCategory.U2R, Severity.CRITICAL, "Mirai botnet GRE-over-IP flood."),
    _e("Mirai-udpplain", "ciciot2023", AttackCategory.U2R, Severity.CRITICAL, "Mirai botnet plain UDP flood."),
]

_INDEX: dict[tuple[str, str], AttackTypeInfo] = {(e.raw_label, e.dataset): e for e in _TAXONOMY}


def get_info(raw_label: str, dataset: str) -> AttackTypeInfo | None:
    return _INDEX.get((raw_label.strip(), dataset))


def get_category(raw_label: str, dataset: str) -> AttackCategory:
    info = get_info(raw_label, dataset)
    return info.category if info else AttackCategory.OTHER


def get_severity(raw_label: str, dataset: str) -> Severity:
    info = get_info(raw_label, dataset)
    if info:
        return info.severity
    return _DEFAULT_SEVERITY_BY_CATEGORY[AttackCategory.OTHER]


def taxonomy_for_dataset(dataset: str) -> list[AttackTypeInfo]:
    return [e for e in _TAXONOMY if e.dataset == dataset]


def all_categories() -> list[AttackCategory]:
    return list(AttackCategory)


def check_coverage(raw_labels: list[str], dataset: str) -> list[str]:
    """
    Given raw label values actually seen in a dataset (e.g. the unique
    values of the Label column), returns any that AREN'T in this taxonomy —
    these silently fall back to category "Other" today and need a manual
    look before final training.
    """
    known = {e.raw_label for e in _TAXONOMY if e.dataset == dataset}
    seen = {str(label).strip() for label in raw_labels}
    return sorted(seen - known)