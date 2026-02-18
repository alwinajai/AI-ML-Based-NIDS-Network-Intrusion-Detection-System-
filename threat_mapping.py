# src/threat_mapping.py
"""
Threat mapping utility.

This module provides:
- a small curated mapping table that maps model prediction labels or textual
  predictions to threat metadata (category, description, example CVEs, CVSS,
  severity).
- function `map_prediction_to_threat(prediction)` which returns a dict with:
    {
        "category": <str>,
        "description": <str>,
        "cves": [<str>, ...],
        "cvss": <float or None>,
        "severity": <"Low"|"Medium"|"High"|"Critical"|"Info">
    }

You can expand the mapping dictionary later by adding more rules or by
integrating external CVE sources (NVD) if you have internet / API access.
"""

from typing import List, Dict, Any
import re

# A minimal curated mapping; expand as required for your dataset.
# Keys can be prediction strings (lowercased) or label numbers as strings.
THREAT_DB = {
    # Example: if model outputs textual label "Attack" or "Malicious"
    "attack": {
        "category": "Generic Network Attack",
        "description": "Traffic flagged as malicious by model heuristics.",
        "cves": [],
        "cvss": None,
        "severity": "High",
    },
    "malicious": {
        "category": "Generic Network Attack",
        "description": "Traffic flagged as malicious by model heuristics.",
        "cves": [],
        "cvss": None,
        "severity": "High",
    },
    # Numeric label examples (if models return 1 for attack)
    "1": {
        "category": "Generic Network Attack",
        "description": "Anomalous packet flow indicative of malicious activity.",
        "cves": [],
        "cvss": None,
        "severity": "High",
    },
    "sql_injection": {
        "category": "SQL Injection",
        "description": "Payload patterns indicate possible SQL injection attempts.",
        "cves": ["CVE-2012-xxxx", "CVE-2013-yyyy"],  # replace with real CVEs if known
        "cvss": 7.5,
        "severity": "High",
    },
    "xss": {
        "category": "Cross-Site Scripting (XSS)",
        "description": "Payload contains script-injection patterns consistent with XSS.",
        "cves": ["CVE-2014-xxxx"],
        "cvss": 6.1,
        "severity": "Medium",
    },
    "ransomware": {
        "category": "Ransomware-related traffic",
        "description": "Payload and destination patterns resemble known ransomware C2 communications.",
        "cves": ["CVE-2019-19781"],
        "cvss": 9.8,
        "severity": "Critical",
    },
    "smb_exploit": {
        "category": "SMB Exploit",
        "description": "Traffic looks like SMB exploit attempts.",
        "cves": ["CVE-2017-0144"],  # e.g., EternalBlue
        "cvss": 8.8,
        "severity": "Critical",
    },
    "port_scan": {
        "category": "Port Scanning",
        "description": "High rate of connection attempts to multiple ports - port scanning.",
        "cves": [],
        "cvss": None,
        "severity": "Low",
    },
    "dos": {
        "category": "Denial of Service (DoS)",
        "description": "Traffic patterns indicate volumetric or application-level DoS.",
        "cves": [],
        "cvss": None,
        "severity": "High",
    },
    "benign": {
        "category": "Benign",
        "description": "Normal/legitimate traffic.",
        "cves": [],
        "cvss": None,
        "severity": "Info",
    },
}

def _normalise_pred(pred) -> str:
    """
    Normalises model prediction into a lookup key:
    - Accepts numbers, strings
    - Lowercases strings and strips whitespace
    - Attempts to remove punctuation for fuzzy matching
    """
    if pred is None:
        return "benign"
    # If numeric label
    try:
        if isinstance(pred, (int, float)):
            # convert to integer string if close to int
            if float(pred).is_integer():
                return str(int(pred))
            return str(pred)
    except Exception:
        pass
    s = str(pred).strip().lower()
    # Remove punctuation but keep underscores and hyphens
    s_clean = re.sub(r"[^\w\-_]+", " ", s).strip()
    # common synonyms mapping
    synonyms = {
        "attack": "attack",
        "malicious": "malicious",
        "benign": "benign",
        "normal": "benign",
        "scan": "port_scan",
        "portscan": "port_scan",
        "port-scan": "port_scan",
        "dos": "dos",
        "ddos": "dos",
        "sql-injection": "sql_injection",
        "sql injection": "sql_injection",
        "xss": "xss",
        "ransom": "ransomware",
        "smb": "smb_exploit",
    }
    # check direct mappings
    if s in THREAT_DB:
        return s
    if s_clean in synonyms:
        return synonyms[s_clean]
    for k in synonyms:
        if s_clean == k:
            return synonyms[k]
    # try token find (e.g., "possible sql injection")
    tokens = s_clean.split()
    for t in tokens:
        if t in synonyms:
            return synonyms[t]
        if t in THREAT_DB:
            return t
    # fallback - return the cleaned string itself
    return s_clean

def map_prediction_to_threat(prediction) -> Dict[str, Any]:
    """
    Convert model prediction into a threat metadata dict with fields:
    category, description, cves (list), cvss (float or None), severity (string).
    """
    key = _normalise_pred(prediction)
    # direct match
    if key in THREAT_DB:
        info = THREAT_DB[key].copy()
        # ensure keys exist
        info.setdefault("cves", [])
        info.setdefault("cvss", None)
        info.setdefault("severity", "Unknown")
        info.setdefault("description", "")
        info.setdefault("category", key if "category" not in info else info["category"])
        return info
    # try search for any known token inside key
    for k in THREAT_DB:
        if k in key:
            info = THREAT_DB[k].copy()
            info.setdefault("cves", [])
            return info
    # fallback - benign-like mapping for unknown values containing 'benign' or 'normal'
    if "benign" in key or "normal" in key:
        return THREAT_DB["benign"].copy()
    # default unknown
    return {
        "category": "Unknown",
        "description": f"No mapped threat metadata for prediction '{prediction}'",
        "cves": [],
        "cvss": None,
        "severity": "Unknown",
    }
