from django.apps import AppConfig


class WellnessConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "wellness"
    verbose_name = "Student Wellness"

    def ready(self):
        # Warm up the ML model once at startup so the first analysis is fast.
        from .ml.predict import get_model  # noqa: F401
        try:
            get_model()
        except Exception:
            # Model files missing -> views will show a friendly error page.
            pass
