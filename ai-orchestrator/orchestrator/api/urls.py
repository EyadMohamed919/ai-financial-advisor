from django.urls import path
from .views import classify_intent_view, classify_transaction_view

urlpatterns = [
    path("classify-intent/", classify_intent_view, name="classify_intent"),
    path("classify-transaction/", classify_transaction_view, name="classify_transaction"),
]