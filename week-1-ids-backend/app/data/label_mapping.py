"""
Maps the many raw attack-name strings in CICIDS2017 / CIC-IoT2023 down to the
5 output categories your proposal's Technical Scope slide commits to:
    Normal | DoS | Probe | R2L | U2R

This mapping is a reasonable first pass based on standard NSL-KDD-style
category definitions, applied to the closest matching attacks in each
dataset. Review it with your team / supervisor before final training —
some CIC-IoT2023 attack types (e.g. Mirai botnet floods) don't map cleanly
onto the classic DoS/Probe/R2L/U2R taxonomy and you may want to keep them
as their own category instead of forcing a fit.
"""

# --- CICIDS2017 ---
# Raw labels as they appear in the official CSVs (after whitespace stripping).
CICIDS2017_LABEL_MAP: dict[str, str] = {
    "BENIGN": "Normal",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "Heartbleed": "DoS",
    "PortScan": "Probe",
    "FTP-Patator": "R2L",
    "SSH-Patator": "R2L",
    "Web Attack \u2013 Brute Force": "R2L",
    "Web Attack \u2013 XSS": "R2L",
    "Web Attack \u2013 Sql Injection": "R2L",
    "Infiltration": "U2R",
    "Bot": "U2R",
}

# --- CIC-IoT2023 ---
# The dataset ships 33 fine-grained attack labels grouped into 7 families.
# Mapped here onto the same 5 output categories; families not listed fall
# back to "Other" (see map_label below) and should be triaged manually.
CIC_IOT2023_LABEL_MAP: dict[str, str] = {
    "BenignTraffic": "Normal",
    # DDoS / DoS family
    "DDoS-RSTFINFlood": "DoS", "DDoS-PSHACK_Flood": "DoS", "DDoS-SYN_Flood": "DoS",
    "DDoS-UDP_Flood": "DoS", "DDoS-TCP_Flood": "DoS", "DDoS-ICMP_Flood": "DoS",
    "DDoS-SynonymousIP_Flood": "DoS", "DDoS-ACK_Fragmentation": "DoS",
    "DDoS-UDP_Fragmentation": "DoS", "DDoS-ICMP_Fragmentation": "DoS",
    "DDoS-SlowLoris": "DoS", "DDoS-HTTP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Reconnaissance / scanning family
    "Recon-PingSweep": "Probe", "Recon-OSScan": "Probe", "Recon-PortScan": "Probe",
    "VulnerabilityScan": "Probe", "Recon-HostDiscovery": "Probe",
    # Web / brute-force / spoofing family -> closest analogue is R2L (remote-to-local)
    "DNS_Spoofing": "R2L", "MITM-ArpSpoofing": "R2L", "BrowserHijacking": "R2L",
    "Backdoor_Malware": "R2L", "CommandInjection": "R2L", "SqlInjection": "R2L",
    "XSS": "R2L", "Uploading_Attack": "R2L", "DictionaryBruteForce": "R2L",
    # Mirai botnet family -> closest analogue is U2R (privilege/host compromise)
    "Mirai-greeth_flood": "U2R", "Mirai-greip_flood": "U2R", "Mirai-udpplain": "U2R",
}


def map_label(raw_label: str, dataset: str) -> str:
    """
    dataset: "cicids2017" or "ciciot2023"
    Falls back to "Other" for anything not in the map so it's visible in
    class-balance reports rather than silently mis-binned.
    """
    table = CICIDS2017_LABEL_MAP if dataset == "cicids2017" else CIC_IOT2023_LABEL_MAP
    return table.get(raw_label.strip(), "Other")