"""
Safety layer — runs BEFORE and INDEPENDENTLY of the ML classifier.

The stress classifier must never be the thing that decides whether a
student is in crisis. These pattern checks catch explicit self-harm /
suicide / immediate-danger language and trigger the safety pathway:
a prominent, calm support card with verified crisis resources and a nudge
to contact emergency services and a trusted person.

This is a keyword safety net, not a clinical tool. It errs on the side of
showing support resources.
"""

import re

# Patterns indicating possible self-harm, suicidal ideation, or immediate danger.
_CRISIS_PATTERNS = [
    r"\bkill\s+myself\b",
    r"\bkilling\s+myself\b",
    r"\bsuicid\w*\b",
    r"\bself[\s-]?harm\b",
    r"\bhurt\s+myself\b",
    r"\bhurting\s+myself\b",
    r"\bcut\s+myself\b",
    r"\bcutting\s+myself\b",
    r"\bend\s+my\s+life\b",
    r"\bend\s+it\s+all\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+i\s+(was|were)\s+dead\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bgive\s+up\s+on\s+life\b",
    r"\boverdose\b",
    r"\bhang\s+myself\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]

# Verified crisis resources (India-first; adapt for other regions).
CRISIS_RESOURCES = [
    {
        "name": "KIRAN Mental Health Helpline (Govt. of India)",
        "contact": "1800-599-0019",
        "detail": "Toll-free, 24x7, 13 languages",
    },
    {
        "name": "Vandrevala Foundation",
        "contact": "1860-2662-345 / 1800-2333-330",
        "detail": "24x7 crisis support across India",
    },
    {
        "name": "AASRA",
        "contact": "9820466726",
        "detail": "24x7 suicide prevention helpline",
    },
    {
        "name": "iCall (TISS)",
        "contact": "9152987821",
        "detail": "Mon–Sat, 10 AM – 8 PM IST",
    },
    {
        "name": "Emergency services (India)",
        "contact": "112",
        "detail": "For immediate danger, call now and reach a trusted person",
    },
]


def detect_crisis(text: str) -> bool:
    """Return True if the text contains crisis-indicating language."""
    if not text:
        return False
    return any(p.search(text) for p in _COMPILED)


def safety_card_context() -> dict:
    return {"crisis_resources": CRISIS_RESOURCES}
