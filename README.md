# 💜 MindWell — AI-Powered Student Stress Detection & Mental Wellness Platform

A modern, calming web platform where students write about what they're feeling,
get a gentle ML-based stress estimate, receive personalized supportive suggestions,
chat with a wellness assistant, and follow a daily wellness plan.

🌐 **Live Demo:** [https://mindwell-ai-one.vercel.app/](https://mindwell-ai-one.vercel.app/)

**Stack:** Django · Bootstrap 5 · scikit-learn (TF-IDF + Logistic Regression) ·
SQLite · Chart.js · optional LLM API for AI-personalized recommendations.

---

## ✨ Features

| Page | What it does |
|---|---|
| **Home** | "Your Mental Wellness Matters" hero, how-it-works, model transparency strip |
| **Dashboard** | Daily motivation, wellness tips, relaxation exercises, quick mood check-in |
| **Write** | Private text box + mood picker → **Analyze My Stress** |
| **Results** | Stress category, compassionate explanation, themes, 4–6 suggestions, breathing exercise, wellness-plan & professional-support actions, crisis safety card when needed |
| **Assistant** | Supportive AI chat (LLM when configured, guided mode otherwise) |
| **My Plan** | Daily wellness plan with tappable checkboxes & progress bar |
| **Resources** | Breathing guides, study/exam tips, sleep hygiene, counselor info, verified helplines, FAQ |
| **History** | Optional private history: 14-day stress trend chart, past check-ins, mood journal, one-tap delete |
| **Model info** | Honest metrics: accuracy, precision/recall/F1, confusion matrix, limitations |

**Safety built in:** crisis-language detection (independent of the classifier) triggers a
support card with helplines; disclaimers everywhere; no diagnoses, ever; tracking is
opt-in, anonymous (session-keyed), and deletable.

---

## 🚀 Quick start (5 minutes)

**Requirements:** Python 3.10+ and `pip`.

```bash
# 1. Unzip and enter the project
cd student-wellness-platform

# 2. Create & activate a virtual environment
python -m venv .venv
# For Windows: .venv\Scripts\activate
# For macOS / Linux: source .venv/bin/activate        
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Model is already trained & included — this step is optional / for retraining)
# python wellness/ml/train.py

# 5. Set up the database
python manage.py migrate

# 6. Run!
python manage.py runserver
```

Open **http://127.0.0.1:8000** in your browser (or visit the live deployment at [**https://mindwell-ai-one.vercel.app/**](https://mindwell-ai-one.vercel.app/)). That's it — no build step, no extra services.

> 🌐 **Live Demo:** You can also test the platform directly without any local setup at [**https://mindwell-ai-one.vercel.app/**](https://mindwell-ai-one.vercel.app/).

---

## 🧠 The ML model

- **Pipeline:** text cleaning → TF-IDF (1–2 grams, 5k features) → Logistic Regression (multinomial, balanced).
- **Data:** `wellness/ml/data/stress_dataset.csv` — 870 labeled student-stress texts
  (low / moderate / high), generated from class-specific template banks. **This is a
  sample dataset** for demonstrating the full pipeline; swap in a real,
  ethically-collected labeled dataset for research use (documented below).
- **Evaluation (honest):** group-aware train/test split (paraphrases of the same
  example never leak across the split) — **94.4% accuracy, macro F1 0.94** on 180
  held-out texts. Full report: run the app and open **Model info**, or read
  `wellness/ml/models/metrics.json`.
- **Files:** `build_dataset.py` (regenerate data) · `train.py` (train & evaluate) ·
  `predict.py` (load & serve) · `models/` (saved vectorizer, model, metrics).

### Retraining / using your own dataset

1. Replace `wellness/ml/data/stress_dataset.csv` with your CSV (`text,label,group`
   columns; `group` links paraphrases of one example so the split stays honest).
2. Run `python wellness/ml/train.py` — it prints precision/recall/F1 + confusion
   matrix and saves fresh artifacts into `wellness/ml/models/`.
3. Restart the server. The app loads the new model automatically.

---

## 🤖 AI recommendations (LLM — optional)

Without any API key, MindWell runs in **Guided support mode**: a built-in,
theme-aware engine writes personalized suggestions from the student's actual
words (labeled "🌿 Guided support" in the UI — never pretending to be AI).

To enable **AI-personalized** suggestions and chat replies:

```bash
# .env (or environment variables)
LLM_API_KEY=sk-your-key
LLM_API_BASE=https://api.openai.com/v1   # any OpenAI-compatible endpoint works
LLM_MODEL=gpt-4o-mini
```

The LLM is called with a strict safety system prompt (supportive, non-diagnostic,
crisis-aware). If the API fails, the app falls back to guided mode automatically.

---

## 🔌 JSON API (REST-style)

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/analyze/` | POST | `{"text": "...", "mood": "anxious"}` | label, confidence, probabilities, themes, recommendations, crisis flag |
| `/api/chat/` | POST | `{"message": "..."}` | reply, source, crisis flag + resources |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Exams next week and I cannot concentrate, feeling overwhelmed."}'
```

---

## 🗂️ Project structure

```
student-wellness-platform/
├── manage.py                    # Django entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template (LLM keys, secrets)
├── mindwell/                    # Project settings (settings.py, urls.py, wsgi.py)
└── wellness/                    # Main app
    ├── models.py                # Assessment, WellnessTask, MoodEntry (anonymous, session-keyed)
    ├── views.py / urls.py       # 8 pages + 2 JSON APIs
    ├── ai.py                    # LLM + guided-mode recommendations & chat
    ├── safety.py                # Crisis-language detection + helplines
    ├── themes.py                # Theme extraction from student text
    ├── ml/
    │   ├── data/build_dataset.py / stress_dataset.csv
    │   ├── train.py             # train + evaluate + save artifacts
    │   ├── predict.py           # load & serve predictions
    │   └── models/              # vectorizer.pkl, model.pkl, metrics.json (pre-trained)
    ├── templates/wellness/      # Bootstrap 5 pages
    ├── static/wellness/         # theme CSS + JS (breathing exercise, chat, charts)
    ├── migrations/              # DB migrations (committed)
    └── tests.py                 # 18 tests: ML, safety, pages, APIs
```

---

## 🧪 Running tests

```bash
python manage.py test wellness
```

Covers: classifier predictions, crisis detection (incl. no false positives),
theme extraction, recommendation engine, every page (200 OK), the full
write→analyze→results→save→delete flow, plan toggling, mood logging, and both APIs.

---

## 🔒 Privacy & safety notes

- No accounts: history is keyed to an anonymous browser session.
- Journal text is stored **only** with explicit opt-in; otherwise only the result is kept.
- One-tap **Delete my data** wipes everything for the session.
- Crisis language triggers the safety pathway **independently** of the ML classifier.
- The app never diagnoses conditions or prescribes anything; disclaimers are shown
  on the results page, footer, and model-info page.
- Do not use student submissions for training without consent (see `train.py`).

---

## 📝 For your project report / viva

- **Problem:** students struggle to recognize and articulate stress; help often comes too late.
- **Solution:** a private, friendly platform combining an ML stress estimator with
  supportive guidance — lowering the barrier to the first step of seeking help.
- **ML contribution:** full pipeline (data → training → honest evaluation →
  deployment), with leakage-free evaluation and published metrics.
- **Engineering:** Django MVT, Bootstrap 5 responsive UI, session-based privacy
  model, REST-style JSON APIs, 18 automated tests.
- **Honest framing:** sample dataset + clearly-labeled guided mode — the project
  demonstrates the *complete working system*, ready to plug in real data / LLM keys.

---

*Built with care for student wellbeing.* 💜
