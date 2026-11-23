from django.urls import path
from .views import (
    classify_intent_view,
    classify_transaction_view,
    extract_transaction_item_view,
    chat_message
)

urlpatterns = [
    path("classify-intent/", classify_intent_view, name="classify_intent"),
    path("classify-transaction/", classify_transaction_view, name="classify_transaction"),
    path("extract-transaction-item/", extract_transaction_item_view, name="extract_expense"),
    path("chat/", chat_message, name="chat_message"),
]