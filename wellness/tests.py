"""Smoke + unit tests for the MindWell platform."""

import json

from django.test import Client, TestCase
from django.urls import reverse

from .ai import chat_reply, generate_recommendations
from .ml.predict import predict
from .models import Assessment, MoodEntry, WellnessTask
from .safety import detect_crisis
from .themes import extract_themes


class MLPipelineTests(TestCase):
    def test_predict_low(self):
        r = predict("I feel calm and relaxed today. My studies are on track and I slept well.")
        self.assertEqual(r["label"], "low")
        self.assertIn("probabilities", r)

    def test_predict_high(self):
        r = predict("I am drowning in work. I cannot sleep, panic attacks every day, I feel hopeless.")
        self.assertEqual(r["label"], "high")

    def test_predict_empty_raises(self):
        with self.assertRaises(ValueError):
            predict("   ")


class SafetyTests(TestCase):
    def test_crisis_detected(self):
        self.assertTrue(detect_crisis("I want to kill myself"))
        self.assertTrue(detect_crisis("I have been thinking about suicide lately"))

    def test_no_false_positive(self):
        self.assertFalse(detect_crisis("I have exams next week and feel stressed"))
        self.assertFalse(detect_crisis(""))


class ThemeTests(TestCase):
    def test_themes_found(self):
        themes = extract_themes("Exams are near and I cannot sleep at night, feeling anxious.")
        self.assertIn("Exam pressure", themes)
        self.assertIn("Sleep difficulties", themes)


class RecommendationTests(TestCase):
    def test_guide_mode_returns_list(self):
        recs, source = generate_recommendations(
            "exams are stressing me out", "moderate", ["Exam pressure"], "anxious")
        self.assertEqual(source, "guide")
        self.assertTrue(3 <= len(recs) <= 6)

    def test_chat_reply_safe(self):
        reply, source = chat_reply("I am stressed about exams", [])
        self.assertTrue(len(reply) > 20)


class PageTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_all_pages_load(self):
        for name in ["home", "dashboard", "write", "assistant", "plan",
                     "resources", "history", "model_info"]:
            resp = self.client.get(reverse(f"wellness:{name}"))
            self.assertEqual(resp.status_code, 200, f"page {name} failed")

    def test_results_redirects_without_analysis(self):
        resp = self.client.get(reverse("wellness:results"))
        self.assertEqual(resp.status_code, 302)

    def test_full_write_analyze_flow(self):
        resp = self.client.post(reverse("wellness:analyze"), {
            "text": "I have exams next week, I cannot concentrate, I feel overwhelmed "
                    "and I am worried about disappointing my parents.",
            "mood": "anxious",
        })
        self.assertEqual(resp.status_code, 302)
        resp = self.client.get(reverse("wellness:results"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "stress")

    def test_short_text_rejected(self):
        resp = self.client.post(reverse("wellness:analyze"), {"text": "hi"})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "little more")

    def test_crisis_flow_shows_safety_card(self):
        self.client.post(reverse("wellness:analyze"), {
            "text": "I feel hopeless and I want to kill myself, everything is too much.",
            "mood": "overwhelmed",
        })
        resp = self.client.get(reverse("wellness:results"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "You don't have to face this alone")

    def test_save_and_delete_history(self):
        self.client.post(reverse("wellness:analyze"), {
            "text": "Feeling calm and good today, studies going well.", "mood": "happy"})
        self.client.post(reverse("wellness:save_assessment"), {"keep_text": "on"})
        self.assertEqual(Assessment.objects.count(), 1)
        a = Assessment.objects.first()
        self.assertTrue(a.text_saved)
        resp = self.client.get(reverse("wellness:history"))
        self.assertContains(resp, "Past check-ins")
        self.client.post(reverse("wellness:delete_data"))
        self.assertEqual(Assessment.objects.count(), 0)

    def test_second_analysis_not_marked_saved_and_saves_fresh(self):
        # regression: a stale "saved" flag from analysis #1 leaked onto
        # analysis #2's results page, and #2 could never be saved
        self.client.post(reverse("wellness:analyze"), {
            "text": "Feeling calm and good today, studies going well.", "mood": "happy"})
        self.client.post(reverse("wellness:save_assessment"), {})
        resp = self.client.get(reverse("wellness:results"))
        self.assertContains(resp, "Saved to your private history")

        # second, distinct analysis
        self.client.post(reverse("wellness:analyze"), {
            "text": "I have exams next week and I cannot concentrate, I feel overwhelmed.",
            "mood": "anxious"})
        resp = self.client.get(reverse("wellness:results"))
        self.assertNotContains(resp, "Saved to your private history")
        self.assertContains(resp, "Keep this in your private history?")

        # saving the second analysis works -> history holds both
        self.client.post(reverse("wellness:save_assessment"), {})
        self.assertEqual(Assessment.objects.count(), 2)

    def test_double_save_creates_no_duplicate(self):
        self.client.post(reverse("wellness:analyze"), {
            "text": "Feeling calm and good today, studies going well.", "mood": "happy"})
        self.client.post(reverse("wellness:save_assessment"), {})
        self.client.post(reverse("wellness:save_assessment"), {})
        self.assertEqual(Assessment.objects.count(), 1)

    def test_plan_tasks_and_toggle(self):
        resp = self.client.get(reverse("wellness:plan"))
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(WellnessTask.objects.exists())
        task = WellnessTask.objects.first()
        resp = self.client.post(reverse("wellness:toggle_task", args=[task.id]),
                                HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(json.loads(resp.content)["done"])

    def test_mood_logging(self):
        self.client.post(reverse("wellness:log_mood"),
                         {"mood": "okay", "next": reverse("wellness:dashboard")})
        self.assertEqual(MoodEntry.objects.count(), 1)

    def test_api_analyze(self):
        resp = self.client.post(
            reverse("wellness:api_analyze"),
            json.dumps({"text": "I feel relaxed and happy, everything is going well today.",
                        "mood": "happy"}),
            content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data["label"], "low")
        self.assertIn("recommendations", data)

    def test_api_chat(self):
        resp = self.client.post(
            reverse("wellness:api_chat"),
            json.dumps({"message": "I am anxious about exams"}),
            content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn("reply", data)
