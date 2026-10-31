from django.contrib import admin

from .models import Assessment, MoodEntry, WellnessTask


@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ("stress_label", "mood", "created_at", "text_saved", "crisis_flag")
    list_filter = ("stress_label", "mood")


@admin.register(WellnessTask)
class WellnessTaskAdmin(admin.ModelAdmin):
    list_display = ("title", "plan_date", "done")


@admin.register(MoodEntry)
class MoodEntryAdmin(admin.ModelAdmin):
    list_display = ("mood", "created_at")
