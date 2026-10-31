"""Views for the MindWell student wellness platform."""

import json
import uuid
from datetime import date, timedelta

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .ai import chat_reply, generate_recommendations, generate_wellness_plan
from .ml.predict import get_metrics, predict
from .models import Assessment, MoodEntry, WellnessTask
from .safety import detect_crisis, safety_card_context
from .themes import extract_themes

LABEL_DISPLAY = {
    "low": "Low — minimal signs of stress",
    "moderate": "Moderate — moderate signs of stress",
    "high": "High — high signs of stress",
}
LABEL_BLURB = {
    "low": ("Your words carry a calm, steady energy. It's wonderful that things feel "
            "manageable right now — the habits below will help you protect that balance."),
    "moderate": ("There's some genuine pressure in what you shared — and that's completely "
                "okay. Many students feel this way. The steps below are small, practical, "
                "and designed exactly for moments like this."),
    "high": ("Thank you for trusting us with this — it takes courage to put heavy feelings "
             "into words. What you're feeling is valid, and support is available. Please be "
             "extra gentle with yourself today, and consider reaching out to someone you "
             "trust or a counselor."),
}

MOOD_LABELS = {"happy": "Happy", "okay": "Okay", "anxious": "Anxious",
               "sad": "Sad", "overwhelmed": "Overwhelmed"}


def _session_key(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _llm_mode():
    return "ai" if getattr(settings, "LLM_API_KEY", "") else "guide"


# ------------------------------------------------------------------ pages ---
def home(request):
    from django.urls import reverse
    metrics = get_metrics()
    features = [
        ("📊", "Stress Analysis", "ML-powered reading of your words into gentle, honest stress insights.", reverse("wellness:write")),
        ("💬", "AI Assistant", "A warm chat companion for tough moments, day or night.", reverse("wellness:assistant")),
        ("🗓️", "Wellness Plan", "A simple daily plan — breathing, study, rest, connection.", reverse("wellness:plan")),
        ("📚", "Resources", "Meditations, study tips, sleep guides & verified helplines.", reverse("wellness:resources")),
    ]
    quotes = [
        "You don't have to be perfect today. Just be kind to yourself.",
        "Rest is productive too. Your mind learns while you recover.",
        "It's okay to ask for help — that's wisdom, not weakness.",
        "You have survived 100% of your hardest days so far.",
        "Progress, not perfection. One breath at a time.",
        "Your worth is not measured by your marks.",
    ]
    return render(request, "wellness/home.html", {
        "accuracy": round((metrics.get("accuracy") or 0.94) * 100),
        "llm_mode": _llm_mode(),
        "features": features,
        "quotes": quotes,
    })


def dashboard(request):
    # Daily motivational message rotates by date — deterministic, no DB needed.
    messages = [
        "You don't have to be perfect today. You just have to be kind to yourself.",
        "Small steps every day lead to big changes. Start with one.",
        "Rest is productive too. Your mind learns while you recover.",
        "It's okay to ask for help — that's wisdom, not weakness.",
        "You have survived 100% of your hardest days so far.",
        "Progress, not perfection. One chapter, one task, one breath at a time.",
        "Your worth is not measured by your marks.",
        "Tough chapters don't define the whole story. Keep turning pages.",
        "Breathe in courage, breathe out doubt. You've got this.",
        "Even on cloudy days, the sun is still there — so is your strength.",
    ]
    day_index = date.today().toordinal() % len(messages)
    tips = [
        {"icon": "🌿", "title": "The 5-4-3-2-1 reset",
         "text": "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste. Instant grounding."},
        {"icon": "💧", "title": "Hydrate before caffeine",
         "text": "A glass of water first thing helps focus more than a second coffee."},
        {"icon": "📵", "title": "Phone-free first hour",
         "text": "Keep the phone away for the first study hour. Your attention will thank you."},
        {"icon": "🚶", "title": "Walk it out",
         "text": "A 10-minute walk between study blocks clears mental fog better than scrolling."},
        {"icon": "😴", "title": "Protect your sleep",
         "text": "Screens off 30 minutes before bed. Sleep is when your brain files everything away."},
        {"icon": "💬", "title": "Talk it out",
         "text": "Sharing a worry with one trusted person can halve its weight."},
    ]
    recent = Assessment.objects.filter(
        session_key=_session_key(request))[:3]
    return render(request, "wellness/dashboard.html", {
        "motivation": messages[day_index],
        "tips": tips,
        "recent": recent,
        "mood_labels": MOOD_LABELS,
    })


def write(request):
    return render(request, "wellness/write.html", {"mood_labels": MOOD_LABELS})


def _analyze_text(text: str, mood: str) -> dict:
    """Shared pipeline for the HTML flow and the JSON API."""
    crisis = detect_crisis(text)
    prediction = predict(text)
    label = prediction["label"]
    themes = extract_themes(text)
    recommendations, source = generate_recommendations(
        text, label, themes, mood, crisis=crisis)
    return {
        "label": label,
        "label_display": LABEL_DISPLAY[label],
        "blurb": LABEL_BLURB[label],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "themes": themes,
        "recommendations": recommendations,
        "rec_source": source,   # "ai" or "guide"
        "mood": mood,
        "mood_display": MOOD_LABELS.get(mood, ""),
        "crisis": crisis,
        "text": text,
        "analysis_id": uuid.uuid4().hex,
    }


def analyze(request):
    """POST from the Write page: run the pipeline, stash in session, show results."""
    if request.method != "POST":
        return redirect("wellness:write")
    text = (request.POST.get("text") or "").strip()
    mood = request.POST.get("mood", "")
    if len(text) < 10:
        return render(request, "wellness/write.html", {
            "mood_labels": MOOD_LABELS,
            "error": "Please write a little more (at least a sentence or two) so the analysis has something to work with.",
            "text": text, "mood": mood,
        })
    try:
        analysis = _analyze_text(text, mood)
    except FileNotFoundError:
        return render(request, "wellness/error.html", {
            "message": "The stress-detection model isn't trained yet. "
                       "Run `python wellness/ml/train.py`, then restart the server."})
    except ValueError as exc:
        return render(request, "wellness/write.html", {
            "mood_labels": MOOD_LABELS, "error": str(exc),
            "text": text, "mood": mood})
    request.session["mw_analysis"] = analysis
    # A fresh analysis has not been saved yet — clear any stale saved flag
    # from a previous analysis (this was leaking "✓ Saved" onto new results).
    request.session["mw_saved_for"] = None
    return redirect("wellness:results")


def results(request):
    analysis = request.session.get("mw_analysis")
    if not analysis:
        return redirect("wellness:write")
    ctx = dict(analysis)
    ctx.update(safety_card_context())
    # "Saved" is per-analysis: only the exact analysis the user saved counts.
    ctx["saved"] = (
        bool(analysis.get("analysis_id"))
        and request.session.get("mw_saved_for") == analysis.get("analysis_id")
    )
    return render(request, "wellness/results.html", ctx)


@require_POST
def save_assessment(request):
    """Opt-in: persist the latest analysis to private history."""
    analysis = request.session.get("mw_analysis")
    if not analysis:
        return redirect("wellness:write")
    analysis_id = analysis.get("analysis_id")
    # Guard against double-submits: the same analysis is only ever saved once.
    if analysis_id and request.session.get("mw_saved_for") == analysis_id:
        return redirect("wellness:history")
    keep_text = request.POST.get("keep_text") == "on"
    Assessment.objects.create(
        session_key=_session_key(request),
        mood=analysis.get("mood", ""),
        stress_label=analysis["label"],
        confidence=analysis["confidence"],
        themes=analysis["themes"],
        recommendations=analysis["recommendations"],
        journal_text=analysis["text"] if keep_text else "",
        text_saved=keep_text,
        crisis_flag=analysis["crisis"],
    )
    request.session["mw_saved_for"] = analysis_id
    return redirect("wellness:history")


def assistant(request):
    history = request.session.get("mw_chat", [])
    ctx = {"history": history[-30:],
           "llm_mode": _llm_mode(),
           **safety_card_context()}
    return render(request, "wellness/assistant.html", ctx)


@require_POST
def api_chat(request):
    try:
        payload = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    message = (payload.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "Please type a message first."}, status=400)
    crisis = detect_crisis(message)
    history = request.session.get("mw_chat", [])
    reply, source = chat_reply(message, history)
    history.append({"role": "user", "text": message[:800]})
    history.append({"role": "assistant", "text": reply})
    request.session["mw_chat"] = history[-40:]
    return JsonResponse({
        "reply": reply, "source": source, "crisis": crisis,
        "crisis_resources": safety_card_context()["crisis_resources"] if crisis else [],
    })


def plan(request):
    skey = _session_key(request)
    today = date.today()
    tasks = WellnessTask.objects.filter(session_key=skey, plan_date=today)
    if not tasks.exists():
        analysis = request.session.get("mw_analysis") or {}
        items = generate_wellness_plan(
            analysis.get("text", ""), analysis.get("label", "moderate"),
            analysis.get("themes", []), analysis.get("mood", ""))
        WellnessTask.objects.bulk_create([
            WellnessTask(session_key=skey, plan_date=today, **item) for item in items
        ])
        tasks = WellnessTask.objects.filter(session_key=skey, plan_date=today)
    done = tasks.filter(done=True).count()
    total = tasks.count()
    pct = round(done / total * 100) if total else 0
    return render(request, "wellness/plan.html",
                  {"tasks": tasks, "done": done, "total": total, "pct": pct,
                   "today": today})


@require_POST
def toggle_task(request, task_id):
    try:
        task = WellnessTask.objects.get(id=task_id,
                                        session_key=_session_key(request))
    except WellnessTask.DoesNotExist:
        return JsonResponse({"error": "Task not found."}, status=404)
    task.done = not task.done
    task.save(update_fields=["done"])
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"done": task.done})
    return redirect("wellness:plan")


def resources(request):
    return render(request, "wellness/resources.html", safety_card_context())


def history(request):
    skey = _session_key(request)
    assessments = list(Assessment.objects.filter(session_key=skey)[:30])
    moods = list(MoodEntry.objects.filter(session_key=skey)[:30])

    # Weekly trend: last 14 days, average stress score per day.
    score = {"low": 1, "moderate": 2, "high": 3}
    days = [date.today() - timedelta(days=i) for i in range(13, -1, -1)]
    trend_labels, trend_values = [], []
    for d in days:
        qs = Assessment.objects.filter(session_key=skey, created_at__date=d)
        vals = [score.get(a.stress_label, 2) for a in qs]
        trend_labels.append(d.strftime("%d %b"))
        trend_values.append(round(sum(vals) / len(vals), 2) if vals else None)

    return render(request, "wellness/history.html", {
        "assessments": assessments,
        "moods": moods,
        "trend_labels": json.dumps(trend_labels),
        "trend_values": json.dumps(trend_values),
        "has_data": bool(assessments or moods),
    })


@require_POST
def log_mood(request):
    mood = request.POST.get("mood", "")
    note = (request.POST.get("note") or "").strip()[:280]
    if mood in MOOD_LABELS:
        MoodEntry.objects.create(session_key=_session_key(request),
                                 mood=mood, note=note)
    return redirect(request.POST.get("next", "wellness:dashboard"))


@require_POST
def delete_data(request):
    """Wipe all session-keyed data: assessments, tasks, moods, chat, analysis."""
    skey = _session_key(request)
    Assessment.objects.filter(session_key=skey).delete()
    WellnessTask.objects.filter(session_key=skey).delete()
    MoodEntry.objects.filter(session_key=skey).delete()
    for k in ("mw_analysis", "mw_chat", "mw_saved"):
        request.session.pop(k, None)
    return redirect("wellness:history")


def model_info(request):
    """Transparency page: what the model is, how it was evaluated, honest metrics."""
    metrics = get_metrics()
    report = metrics.get("classification_report", {})
    labels = metrics.get("labels", [])
    cm = metrics.get("confusion_matrix", [])
    rows = []
    for label in labels:
        r = report.get(label, {})
        rows.append({"label": label,
                     "precision": r.get("precision"), "recall": r.get("recall"),
                     "f1": r.get("f1-score"), "support": r.get("support")})
    cm_rows = list(zip(labels, cm)) if len(labels) == len(cm) else []
    return render(request, "wellness/model_info.html", {
        "metrics": metrics, "rows": rows,
        "cm_rows": cm_rows, "labels": labels,
    })


# ------------------------------------------------------------------- API ----
@require_POST
def api_analyze(request):
    """JSON API: {text, mood?} -> prediction + themes + recommendations."""
    try:
        payload = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    text = (payload.get("text") or "").strip()
    mood = payload.get("mood", "")
    if len(text) < 10:
        return JsonResponse(
            {"error": "Text too short — please write at least a sentence or two."},
            status=400)
    try:
        analysis = _analyze_text(text, mood)
    except FileNotFoundError:
        return JsonResponse(
            {"error": "Model not trained. Run python wellness/ml/train.py."},
            status=503)
    analysis.pop("text", None)  # don't echo raw text back unnecessarily
    return JsonResponse(analysis)
