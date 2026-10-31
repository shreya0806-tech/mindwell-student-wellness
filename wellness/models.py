"""
Anonymous, consent-based tracking models.

No user accounts: everything is keyed to the Django session key, so history
stays on the student's own browser. Raw journal text is stored ONLY when the
student explicitly opts in ("save my entry"); otherwise only the aggregate
result (label, mood, themes) is kept. The "Delete my data" action wipes all
rows for the session.
"""

from django.db import models


class Assessment(models.Model):
    LABEL_CHOICES = [
        ("low", "Low — minimal signs of stress"),
        ("moderate", "Moderate — moderate signs of stress"),
        ("high", "High — high signs of stress"),
    ]
    MOOD_CHOICES = [
        ("happy", "Happy"),
        ("okay", "Okay"),
        ("anxious", "Anxious"),
        ("sad", "Sad"),
        ("overwhelmed", "Overwhelmed"),
    ]

    session_key = models.CharField(max_length=64, db_index=True)
    mood = models.CharField(max_length=16, choices=MOOD_CHOICES, blank=True)
    stress_label = models.CharField(max_length=16, choices=LABEL_CHOICES)
    confidence = models.FloatField(default=0.0)
    themes = models.JSONField(default=list)          # e.g. ["exam pressure", ...]
    recommendations = models.JSONField(default=list) # snapshot of suggestions
    journal_text = models.TextField(blank=True)      # ONLY with explicit consent
    text_saved = models.BooleanField(default=False)
    crisis_flag = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.stress_label} ({self.created_at:%Y-%m-%d})"


class WellnessTask(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    plan_date = models.DateField(db_index=True)
    title = models.CharField(max_length=200)
    detail = models.CharField(max_length=300, blank=True)
    category = models.CharField(max_length=32, default="wellness")
    done = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["category", "id"]

    def __str__(self):
        return f"[{'x' if self.done else ' '}] {self.title}"


class MoodEntry(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    mood = models.CharField(max_length=16)
    note = models.CharField(max_length=280, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Mood entries"
