"""
Theme extraction — surface the main topics in the student's text so the
results page can say *what* the stress seems to be about (exam pressure,
sleep, loneliness...), grounding the recommendations in their actual words.
"""

import re

THEMES = [
    ("Exam pressure",
     [r"\bexams?\b", r"\btests?\b", r"\bviva\b", r"\bmidterms?\b", r"\bmarks?\b",
      r"\bgrades?\b", r"\bresults?\b", r"\bquestion papers?\b", r"\bsyllabus\b"]),
    ("Study overload",
     [r"\bassignments?\b", r"\bhomework\b", r"\bdeadlines?\b", r"\bsubmissions?\b",
      r"\bprojects?\b", r"\bchapters?\b", r"\bstudy\b", r"\brevisions?\b",
      r"\bnotes\b", r"\bworkload\b"]),
    ("Sleep difficulties",
     [r"\bsleep\b", r"\binsomnia\b", r"\bnightmares?\b", r"\bawake\b",
      r"\bsleepless\b", r"\btired\b", r"\bexhausted\b", r"\bfatigue\b"]),
    ("Anxiety & panic",
     [r"\banxi\w*\b", r"\bpanic\b", r"\bworry\b", r"\bworried\b",
      r"\bfear\b", r"\bscared\b", r"\bnervous\b", r"\bracing heart\b",
      r"\bcan'?t breathe\b"]),
    ("Feeling overwhelmed",
     [r"\boverwhelm\w*\b", r"\btoo much\b", r"\bdrowning\b", r"\bburnout\b",
      r"\bburden\b", r"\bpressure\b", r"\bstressed\b", r"\bstressful\b"]),
    ("Low mood & hopelessness",
     [r"\bsad\b", r"\bcry\w*\b", r"\btears\b", r"\bhopeless\b",
      r"\bworthless\b", r"\bfailure\b", r"\bnumb\b", r"\bempty\b",
      r"\bgive up\b", r"\bpointless\b"]),
    ("Loneliness & relationships",
     [r"\blonely\b", r"\balone\b", r"\bfriends?\b", r"\broommates?\b",
      r"\brelationships?\b", r"\bmiss home\b", r"\bhomesick\b",
      r"\barguments?\b", r"\bfights?\b"]),
    ("Family expectations",
     [r"\bparents?\b", r"\bfamily\b", r"\bdisappoint\b", r"\bexpectations?\b",
      r"\bmother\b", r"\bfather\b"]),
    ("Concentration issues",
     [r"\bconcentrat\w*\b", r"\bfocus\b", r"\bdistract\w*\b",
      r"\bprocrastinat\w*\b", r"\bmemory\b", r"\bforget\b"]),
    ("Physical symptoms",
     [r"\bheadaches?\b", r"\bstomach\b", r"\bchest\b", r"\bdizz\w*\b",
      r"\bappetite\b", r"\bshak\w*\b", r"\bsweat\w*\b", r"\bpains?\b"]),
    ("Future & career worries",
     [r"\bfuture\b", r"\bcareers?\b", r"\binternships?\b", r"\bplacements?\b",
      r"\bjobs?\b", r"\binterviews?\b"]),
    ("Time management",
     [r"\btime\b", r"\blate\b", r"\bbehind\b", r"\bschedule\b",
      r"\bmanage\b", r"\bplann\w*\b"]),
]

_COMPILED = [(name, [re.compile(p, re.IGNORECASE) for p in pats])
             for name, pats in THEMES]


def extract_themes(text: str, max_themes: int = 4) -> list:
    """Return up to `max_themes` theme names found in the text, ordered by hits."""
    if not text:
        return []
    scored = []
    for name, patterns in _COMPILED:
        hits = sum(1 for p in patterns if p.search(text))
        if hits:
            scored.append((hits, name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:max_themes]]
