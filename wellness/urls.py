from django.urls import path

from . import views

app_name = "wellness"

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("write/", views.write, name="write"),
    path("analyze/", views.analyze, name="analyze"),
    path("results/", views.results, name="results"),
    path("results/save/", views.save_assessment, name="save_assessment"),
    path("assistant/", views.assistant, name="assistant"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/analyze/", views.api_analyze, name="api_analyze"),
    path("plan/", views.plan, name="plan"),
    path("plan/toggle/<int:task_id>/", views.toggle_task, name="toggle_task"),
    path("resources/", views.resources, name="resources"),
    path("history/", views.history, name="history"),
    path("history/delete/", views.delete_data, name="delete_data"),
    path("mood/", views.log_mood, name="log_mood"),
    path("model-info/", views.model_info, name="model_info"),
]
