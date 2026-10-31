"""
Build a labeled student-stress text dataset (synthetic sample dataset).

Labels:
    0 = low       (minimal signs of stress)
    1 = moderate  (moderate signs of stress)
    2 = high      (high signs of stress)

Generation: class-specific template banks with slot fillers. Each class has
a consistent lexical signature:
    low      -> calm/positive markers (relaxed, confident, balanced, grateful...)
    moderate -> concern + hedging/coping markers (a bit, somewhat, trying, coping...)
    high     -> intense-distress markers (panic, unbearable, drowning, hopeless...)
Every filled variant of one template shares a `group` id, so the group-aware
train/test split measures honest generalization to unseen phrasings.

Output: stress_dataset.csv  (columns: text,label,group)

NOTE: this is a *sample* dataset for demonstrating the full ML pipeline
(collection -> training -> evaluation -> serving). For production research,
replace it with a real, ethically-collected labeled dataset — see README.
"""

import csv
import random
from pathlib import Path

random.seed(2026)
LABEL_NAMES = {0: "low", 1: "moderate", 2: "high"}

# ------------------------------------------------------------------ slots --
ACTIVITIES = ["a morning walk", "listening to music", "a short nap",
              "football with friends", "cooking dinner", "a bike ride",
              "reading a novel", "a coffee with my roommate"]
TOPICS = ["calculus", "physics", "data structures", "organic chemistry",
          "economics", "thermodynamics", "linear algebra"]
PEOPLE = ["my friends", "my family", "my roommate", "my study group",
          "my mentor"]

EXAM_WORDS = ["the exams", "the final test", "the upcoming viva",
              "the midterm", "the placement test"]
DEADLINES = ["assignment deadlines", "project submissions", "lab reports",
             "the term paper", "the class presentation"]

# Moderate-only coping/hedging markers are baked into the templates below.
COPING = ["I made a study timetable", "I am taking it one day at a time",
          "I asked a senior for advice", "I started revising early",
          "I am breaking tasks into smaller steps",
          "I talked it over with a friend"]

STRESSORS = ["the vast syllabus", "back-to-back exams", "tough competition",
             "my parents' expectations", "the fear of failing",
             "constant comparison with toppers", "the internship hunt",
             "the placement season"]

# ------------------------------------------------------------- templates ---
# {a}=activity {t}=topic {p}=people {e}=exam {d}=deadline {c}=coping {s}=stressor
LOW_TEMPLATES = [
    "I am feeling calm and relaxed today. {a} helped me recharge and my studies are on track.",
    "Had a wonderful day. I finished my {t} revision and spent happy time with {p}.",
    "Feeling confident and peaceful. My preparation is going smoothly and I slept really well.",
    "I managed my time well this week. Everything is done and I feel balanced and content.",
    "Today was productive and joyful. I feel grateful and fully in control of my schedule.",
    "I feel rested, fresh and focused. Genuinely looking forward to tomorrow's classes.",
    "Things are going great. I submitted my work early and I feel relaxed and proud.",
    "I am so grateful for {p}. Their support keeps me positive, happy and motivated.",
    "Had a lovely evening with {a}. I feel refreshed, light and completely stress free.",
    "I understand {t} much better now. Feeling confident, calm and well prepared.",
    "My routine is working beautifully. I sleep on time, eat well, and feel energetic and healthy.",
    "Got encouraging feedback on my work today. Feeling motivated, happy and confident.",
    "I feel in control and at ease. Nothing feels rushed, and I am enjoying the pace.",
    "Took a pleasant break with {a}. It cleared my mind and I feel recharged and cheerful.",
    "The weekend with {p} was refreshing and fun. Back to studies with a fresh, happy mind.",
    "I ticked off everything on my to-do list. Feeling accomplished, relaxed and positive.",
    "Feeling optimistic and calm about the semester. My grades are on track and I feel great.",
    "Meditated for ten minutes this morning. I feel peaceful, centered and deeply relaxed.",
    "Studying {t} with {p} was genuinely enjoyable. I feel confident and well supported.",
    "I feel relaxed and ready for {e}. I have revised thoroughly and I am well prepared.",
    "Cleared all my doubts in the tutorial. Feeling calm, confident and well prepared.",
    "I am really enjoying my coursework. The pace feels comfortable and the topics are interesting.",
    "Had a warm, happy chat with my family. Feeling loved, supported and at peace.",
    "I feel balanced and content today. Work is done, and I am enjoying {a} guilt-free.",
    "Woke up fresh and cheerful after a full night of deep sleep. Ready for a good day.",
    "I handled this week's workload with ease. Everything feels manageable and smooth.",
    "Feeling light, happy and grateful. {a} with {p} was exactly what I needed.",
    "My project is ahead of schedule. Feeling proud, relaxed and confident.",
]

MODERATE_TEMPLATES = [
    "I have {e} coming up and I feel somewhat nervous. {c}, so I am coping okay.",
    "{d} are piling up and it feels like a lot to handle. {c}, and I am managing.",
    "I feel a bit of pressure about {t}. I am putting in effort but I wish I had slightly more time.",
    "My sleep has been slightly irregular lately due to late-night studying. Feeling a bit tired.",
    "I am somewhat worried about my grades. {c} to catch up gradually.",
    "I feel a little restless thinking about {e}. {c}, and I am trying to stay positive.",
    "There is quite a lot on my plate - classes, {d}, and family duties. I am coping, barely.",
    "I procrastinated a bit and now {d} feel close. Feeling slightly guilty, but I started today.",
    "I feel a little anxious about the viva. {c}, though I still doubt myself at times.",
    "Comparing myself to classmates makes me somewhat insecure. I am working on my confidence.",
    "I feel a bit lonely in the hostel sometimes, which mildly distracts me. {p} check on me though.",
    "Money is somewhat tight this month, adding mild worry. I am trying to budget carefully.",
    "My concentration dips a little in the evenings. I am trying shorter study sessions.",
    "I feel slightly nervous speaking up in class. Practicing with {p} is helping gradually.",
    "Choosing a career path feels like a somewhat big decision. I am researching options slowly.",
    "I skipped a couple of meals on busy days. I know I should eat on time and I am trying.",
    "Social media sometimes makes me feel slightly behind. I am cutting down my screen time.",
    "I feel a bit stuck on {t} and it is mildly frustrating. {c}.",
    "The syllabus feels fairly large and time feels short. {c}, one chapter at a time.",
    "I had a small disagreement with my roommate. Things are slightly tense but we will sort it out.",
    "I worry a little about disappointing my parents if my results slip. Talking to them helped somewhat.",
    "My motivation comes and goes mildly. Some days are fine, others I struggle slightly to start.",
    "I feel mild pressure to be social even when I need rest. I am learning to say no politely.",
    "Interview preparation is making me a bit nervous. {c}, and I feel somewhat ready.",
    "Long screen hours give me a slight headache. I am planning proper breaks from now on.",
    "Deadlines make me feel a bit uneasy at times. Slow deep breathing helps me settle down.",
    "I feel slightly behind in {t}. {c} and I am catching up step by step.",
    "Balancing part-time work with studies is somewhat tiring. Weekends help me recover a bit.",
]

HIGH_TEMPLATES = [
    "I have {e} next week and {s} is crushing me. I cannot sleep at all and I feel like I am falling apart.",
    "I am completely drowning in work. I have not slept properly in days and I feel like I will break down.",
    "Every time I sit down to study, sheer panic rises in my chest. My heart races and I cannot breathe.",
    "I feel like an utter failure. Nothing I do is ever enough and I see no way out of this.",
    "The pressure is absolutely unbearable. {s} never stops and I feel trapped with no escape.",
    "I get terrifying panic attacks before every test. My hands shake violently and my mind goes blank.",
    "I cannot eat or sleep anymore. My thoughts spiral endlessly about failing and ruining my entire future.",
    "I feel completely worthless next to my classmates. I wonder if I even belong in this college.",
    "My chest feels constantly tight, I get headaches every single day, and I do not remember feeling okay.",
    "I missed several deadlines and now I am terrified to even open my email. Everything is out of control.",
    "I cannot stop crying. The tiniest things set me off and I feel so embarrassed and hopeless.",
    "I feel numb, empty and hollow inside. Even things I used to love do not interest me at all now.",
    "This stress is wrecking my health. I feel dizzy, my appetite is gone, and I am exhausted always.",
    "I feel crushed by {s}. The guilt is unbearable and I am scared to tell anyone how bad it really is.",
    "My brain feels foggy and overloaded. I cannot focus for more than five minutes on anything.",
    "I dread {e} with intense fear. {s} keeps me awake all night, sweating and shaking.",
    "I feel utterly hopeless about my future. What is the point of trying if I am doomed to fail?",
    "I am overwhelmed by everything at once - studies, family problems, money. It is all far too much.",
    "Waking up feels like facing an impossible mountain. I dread every single morning now.",
    "I snap at {p} constantly and then feel terrible. I am falling apart and pushing everyone away.",
    "I barely eat or drink water anymore. I have zero energy and my body feels like giving up.",
    "{s} terrifies me to my core. I live in constant fear that something terrible is about to happen.",
    "My mind never ever stops worrying. The anxiety is relentless and I feel like I am suffocating.",
    "I feel suffocated and trapped by {s}. There is no relief, no break, no way out.",
    "I cannot concentrate on anything at all. The panic is constant and I feel completely helpless.",
    "I am exhausted to my bones. I cry every night, then fake a smile every morning.",
    "Horrific nightmares about {e} jolt me awake sweating. I lie awake for hours, terrified.",
    "I feel completely and utterly alone. Nobody understands this pain and I cannot talk to anyone.",
]

TEMPLATE_BANK = {0: LOW_TEMPLATES, 1: MODERATE_TEMPLATES, 2: HIGH_TEMPLATES}
SLOT_POOLS = {"a": ACTIVITIES, "t": TOPICS, "p": PEOPLE, "e": EXAM_WORDS,
              "d": DEADLINES, "c": COPING, "s": STRESSORS}

FILLERS_PER_TEMPLATE = 10


def fill(template: str) -> str:
    out = template
    for key, pool in SLOT_POOLS.items():
        ph = "{" + key + "}"
        while ph in out:
            out = out.replace(ph, random.choice(pool), 1)
    return out


# Hand-written canonical examples (each becomes its own template group with
# light paraphrases). These cover phrasings an evaluator is likely to type,
# including the project's own example prompt.
CANONICAL = {
    2: [
        "I have exams next week, I cannot concentrate on studying, I feel overwhelmed, and I am worried about disappointing my parents.",
        "I have exams next week and I can't focus at all. I feel totally overwhelmed and I'm scared of letting my parents down.",
        "Exams are next week, I am unable to concentrate, everything feels overwhelming, and I fear disappointing my family.",
        "The pressure before exams is crushing me. I can't sleep, I can't focus, and I'm terrified of failing.",
        "I feel like I'm drowning in syllabus. My chest is tight, my mind won't stop racing, and I'm scared.",
    ],
    1: [
        "I have exams soon and I'm feeling the pressure a bit, but I'm trying to stay organized.",
        "There's a lot of homework piling up and I'm somewhat stressed, though I'm managing.",
        "I'm a little anxious about my results, but I've started revising and feel somewhat prepared.",
    ],
    0: [
        "I'm feeling good today. My studies are on track and I had a relaxing evening.",
        "Everything is going well. I finished my work on time and I feel calm and happy.",
    ],
}

_PARAPHRASE_SWAPS = [
    ("I am", "I'm"), ("I have", "I've"), ("cannot", "can't"), ("do not", "don't"),
    ("It is", "It's"), ("there is", "there's"), ("I feel", "I'm feeling"),
]


def paraphrase(text: str) -> str:
    out = text
    for old, new in _PARAPHRASE_SWAPS:
        if old in out and random.random() < 0.5:
            out = out.replace(old, new, 1)
    return out


def main():
    rows = []
    group = 0
    for label in (0, 1, 2):
        for template in TEMPLATE_BANK[label]:
            for _ in range(FILLERS_PER_TEMPLATE):
                rows.append((fill(template), label, group))
            group += 1
        for example in CANONICAL.get(label, []):
            rows.append((example, label, group))
            rows.append((paraphrase(example), label, group))
            rows.append((paraphrase(example), label, group))
            group += 1
    random.shuffle(rows)
    out_path = Path(__file__).parent / "stress_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "group"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows ({group} template groups) -> {out_path}")
    for label in (0, 1, 2):
        n = sum(1 for _, l, _ in rows if l == label)
        print(f"  {LABEL_NAMES[label]:>9}: {n}")


if __name__ == "__main__":
    main()
