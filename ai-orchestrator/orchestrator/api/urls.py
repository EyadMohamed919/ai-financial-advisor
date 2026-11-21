from django.urls import path
from .views import classify_intent_view

urlpatterns = [
    path("classify-intent/", classify_intent_view, name="classify_intent"),
]