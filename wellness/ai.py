"""
Personalized recommendations engine.

Two modes:
  1. AI mode — when LLM_API_KEY is configured, an OpenAI-compatible chat
     API generates empathetic, personalized suggestions with a strict
     safety system prompt (supportive, non-diagnostic, crisis-aware).
  2. Guide mode (offline default) — a built-in, theme-aware recommendation
     bank. Every response is clearly labeled in the UI so the app never
     pretends to be something it is not.

Both modes obey the same safety rules: no diagnoses, no medication advice,
professional help is suggested for persistent/severe distress, and crisis
language always triggers the safety pathway (handled in views via safety.py).
"""

import json
import logging
import urllib.request

from django.conf import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MindWell, a warm, non-judgmental student wellness companion.
A student has shared how they feel, and a simple text classifier estimated their
stress level. Your job: give 4-5 short, practical, compassionate suggestions.

STRICT RULES:
- Be supportive and non-judgmental. Validate their feelings first.
- Base suggestions on the student's ACTUAL concerns described in their text.
- Keep each suggestion to 1-2 sentences, easy to follow, realistic for a student.
- Include one short calming technique (e.g. breathing) where fitting.
- If distress sounds persistent or severe, gently suggest talking to a trusted
  person and a qualified counselor / mental health professional.
- NEVER diagnose any mental health condition. NEVER prescribe medication.
- NEVER claim meditation or breathing alone fixes everything.
- NEVER present the stress estimate as a medical fact; it is a rough estimate.
- If the text hints at self-harm or suicide, respond with care, encourage
  contacting emergency services and a trusted person immediately, and share
  that professional help matters. Do not attempt counseling beyond that.
- Reply ONLY with a JSON array of strings, e.g. ["...", "..."]. No other text."""

CHAT_SYSTEM_PROMPT = """You are MindWell, a warm, non-judgmental student wellness companion chatting
with a college student. Keep replies short (2-4 sentences), supportive, and practical.
Suggest small coping steps, breathing exercises, or talking to trusted people/counselors
when fitting. STRICT RULES: never diagnose conditions, never prescribe medication,
never present yourself as a medical professional, never claim breathing/meditation fixes
everything. If the student mentions self-harm, suicide, or immediate danger, respond with
care and urge them to contact emergency services and a trusted person right away."""


def _llm_available() -> bool:
    return bool(getattr(settings, "LLM_API_KEY", ""))


def _call_llm(messages: list, max_tokens: int = 600) -> str:
    """Call an OpenAI-compatible chat completions endpoint. Raises on failure."""
    payload = json.dumps({
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{settings.LLM_API_BASE}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


# ------------------------------------------------------------ guide mode ---
def _theme_tip(theme: str) -> str:
    bank = {
        "Exam pressure": "Break study material into 25-minute focused blocks with 5-minute breaks — small wins calm exam nerves faster than marathon sessions.",
        "Study overload": "List every pending task, then pick just the next 3. Crossing them off one by one turns a mountain back into steps.",
        "Sleep difficulties": "Start a 30-minute wind-down: dim lights, no phone, slow breathing. A rested brain remembers far more than a tired one.",
        "Anxiety & panic": "Try box breathing: inhale 4 counts, hold 4, exhale 4, hold 4. Repeat for 2 minutes whenever the worry spikes.",
        "Feeling overwhelmed": "Pause and name 3 things you CAN control today. Shrink the day down to what's actually in front of you.",
        "Low mood & hopelessness": "Do one tiny kind thing for yourself today — a walk, a favorite song, sunlight. If the heaviness stays for weeks, please talk to a counselor.",
        "Loneliness & relationships": "Reach out to one person you trust today, even with a small message. Connection is a genuine stress reliever.",
        "Family expectations": "Consider writing down what YOU want, separate from others' hopes. A calm, honest conversation with family often helps more than silent pressure.",
        "Concentration issues": "Study in short sprints with the phone in another room. If focus keeps slipping for days, a counselor can help check what's behind it.",
        "Physical symptoms": "Stress lives in the body too — hydrate, eat something nourishing, stretch your neck and shoulders. If symptoms persist, please see a doctor.",
        "Future & career worries": "You don't need the whole 10-year plan today — just the next small step, like one skill or one conversation with a senior.",
        "Time management": "Time-block tomorrow tonight: study, rest, meals, sleep. A plan on paper quiets a noisy mind.",
    }
    return bank.get(theme, "")


_MOOD_OPENERS = {
    "happy": "It's lovely that you're feeling good — let's keep that momentum going.",
    "okay": "Okay is a perfectly fine place to be. A few small habits can keep it steady.",
    "anxious": "Feeling anxious makes complete sense with everything on your plate. Let's soften it together.",
    "sad": "I'm sorry you're feeling low. You deserve support, and small steps can help.",
    "overwhelmed": "When everything feels like too much, we go smaller — one gentle step at a time.",
}

_LABEL_CLOSERS = {
    "low": "Your words show mostly calm, steady energy. These habits will help protect it.",
    "moderate": "There's some real pressure in your words, and that's okay — these steps are designed exactly for this level.",
    "high": "Your words carry a lot of weight right now. Please be extra gentle with yourself — and consider talking to someone you trust or a counselor soon.",
}


def guide_recommendations(text, label, themes, mood) -> list:
    """Theme-aware, deterministic recommendations (offline guide mode)."""
    recs = []
    opener = _MOOD_OPENERS.get(mood, "")
    if opener:
        recs.append(opener)

    seen = set()
    for theme in (themes or [])[:3]:
        tip = _theme_tip(theme)
        if tip and tip not in seen:
            recs.append(tip)
            seen.add(tip)

    # Always-useful anchors, chosen by label intensity.
    anchors = {
        "low": [
            "Keep your sleep window consistent — it's the cheapest performance enhancer there is.",
            "Share one thing you're grateful for today with a friend; it strengthens both of you.",
        ],
        "moderate": [
            "Take a real 10-minute break every hour of studying: stand, stretch, look far away.",
            "Write tonight's top 3 priorities before bed so your mind can switch off.",
        ],
        "high": [
            "Right now, shrink today to the single most important task — everything else can wait.",
            "If this heaviness has lasted more than two weeks, please reach out to your campus counselor or a mental health professional. You don't have to carry this alone.",
        ],
    }
    for tip in anchors.get(label, anchors["moderate"]):
        if tip not in seen:
            recs.append(tip)
            seen.add(tip)

    recs.append(_LABEL_CLOSERS.get(label, ""))
    # 4-5 focused suggestions (plus opener/closer framing).
    return [r for r in recs if r][:6]


def generate_recommendations(text, label, themes, mood, crisis=False) -> tuple:
    """
    Returns (recommendations: list[str], source: "ai" | "guide").
    Falls back to guide mode if the LLM is unavailable or errors.
    """
    if _llm_available():
        try:
            user_msg = (
                f"Stress estimate: {label}. Mood selected: {mood or 'not given'}. "
                f"Main themes: {', '.join(themes) if themes else 'general stress'}. "
                f"Student's words: \"{text[:1200]}\""
                + (" The text contains possible self-harm language — respond with extra care per your rules." if crisis else "")
            )
            raw = _call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            recs = json.loads(raw)
            if isinstance(recs, list) and recs:
                return [str(r)[:400] for r in recs[:6]], "ai"
        except Exception as exc:  # never break the flow; fall back gracefully
            logger.warning("LLM recommendation failed, using guide mode: %s", exc)
    return guide_recommendations(text, label, themes, mood), "guide"


def generate_wellness_plan(text, label, themes, mood) -> list:
    """A simple daily plan: list of {title, detail, category} dicts."""
    plan = [
        {"title": "2-minute box breathing",
         "detail": "Inhale 4 counts, hold 4, exhale 4, hold 4. Repeat 4 rounds.",
         "category": "Calm"},
        {"title": "Study in 3 focused sprints",
         "detail": "25 minutes study + 5 minutes break, phone away. Just 3 sprints.",
         "category": "Study"},
        {"title": "Eat + hydrate on time",
         "detail": "A proper meal and 6-8 glasses of water through the day.",
         "category": "Body"},
        {"title": "7-8 hours of sleep",
         "detail": "Screens off 30 minutes before bed; fixed sleep time.",
         "category": "Body"},
        {"title": "2-line journal",
         "detail": "Write one feeling and one small win from today.",
         "category": "Reflect"},
        {"title": "Connect with someone",
         "detail": "Message or meet one trusted person — friend, family, mentor.",
         "category": "Connect"},
    ]
    if label == "high":
        plan.insert(1, {
            "title": "One gentle priority",
            "detail": "Pick ONLY today's most important task. The rest can wait.",
            "category": "Study"})
        plan.append({
            "title": "Talk to someone you trust",
            "detail": "Share how you feel with a friend, teacher, or counselor.",
            "category": "Connect"})
    if "Sleep difficulties" in (themes or []):
        plan.append({
            "title": "Wind-down routine",
            "detail": "Dim lights, light stretching, no phone for 30 minutes before sleep.",
            "category": "Body"})
    return plan


# --------------------------------------------------------------- chat ------
def chat_reply(message: str, history: list) -> tuple:
    """Return (reply_text, source). History: list of {role, text} dicts."""
    if _llm_available():
        try:
            msgs = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
            for h in history[-8:]:
                msgs.append({"role": "user" if h["role"] == "user" else "assistant",
                             "content": h["text"][:800]})
            msgs.append({"role": "user", "content": message[:800]})
            return _call_llm(msgs, max_tokens=300), "ai"
        except Exception as exc:
            logger.warning("LLM chat failed, using guide mode: %s", exc)

    # Guide-mode conversational fallback: supportive, keyword-aware, safe.
    low = message.lower()
    if any(w in low for w in ("breath", "panic", "anxious", "anxiety", "calm")):
        reply = ("Let's slow things down together. Try this: breathe in for 4 counts, "
                 "hold for 4, breathe out for 6. Do 4 rounds and notice your shoulders "
                 "drop. Would you like a full 2-minute guided session?")
    elif any(w in low for w in ("exam", "study", "syllabus", "marks", "grade", "test")):
        reply = ("Exam pressure is real, and it helps to shrink it. Pick one small topic "
                 "and study it for just 25 minutes — then take a 5-minute break. Small "
                 "wins build momentum. What's the one topic weighing on you most?")
    elif any(w in low for w in ("sleep", "tired", "insomnia", "awake")):
        reply = ("Sleep troubles make everything harder — you're not imagining it. Tonight, "
                 "try screens off 30 minutes before bed and 4-7-8 breathing (in 4, hold 7, "
                 "out 8). A consistent sleep time matters more than total hours.")
    elif any(w in low for w in ("lonely", "alone", "friend", "nobody")):
        reply = ("Feeling alone in this is painful, and it makes sense you'd feel it. "
                 "Is there one person — a friend, senior, or family member — you could "
                 "share even a little of this with today?")
    elif any(w in low for w in ("sad", "cry", "hopeless", "worthless", "failure")):
        reply = ("I'm really sorry you're feeling this way. Those feelings are heavy, and "
                 "you don't have to carry them alone. If this heaviness stays for weeks, "
                 "please consider talking to a counselor — reaching out is a strong step, "
                 "not a weak one.")
    elif any(w in low for w in ("thank", "thanks", "better", "good")):
        reply = ("I'm glad to hear that. Keep being kind to yourself — you're doing better "
                 "than you think. I'm here whenever you need to talk.")
    elif "hello" in low or "hi" == low.strip() or low.strip().startswith("hey"):
        reply = ("Hello, I'm glad you're here. This is a safe, judgment-free space. "
                 "Tell me what's on your mind — studies, sleep, friendships, anything.")
    else:
        reply = ("Thank you for trusting me with that. It sounds like a lot to hold. "
                 "Can you tell me a little more about the part that feels heaviest right "
                 "now? We'll take it one small step at a time.")
    return reply, "guide"
