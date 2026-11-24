import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from api.services.intent_service import predict_intent
from api.services.transaction_type_service import predict_transaction
from api.services.transaction_item_extraction import extract_expense
from api.services.income_service import predict_income
from api.services.expense_service import predict_expense


@csrf_exempt
@require_POST
def chat_message(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message = (data.get("text") or data.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "Field 'text' is required."}, status=400)

    intent = predict_intent(message)
    if intent["intent"] == "transaction":

        transaction = predict_transaction(message)
        tx_type = transaction.get("transaction_type")
        
        if tx_type == "Income":
            category = predict_income(message)
        elif tx_type == "Expense":
            category = predict_expense(message)
        else:
            category = {"category": "failed to classify", "tx_type": tx_type}
        
        item = extract_expense(message)
        ai_message = "I saved your " + item["item"] + " and classified it as " + category["category"] + ". This will be added to your " + tx_type + "s."
        respond = {"respond":ai_message}
        combinedResult = intent | transaction | category | item | respond
    else:
        return JsonResponse({"error": "Couldn't classify the message any of the following: transaction"}, status=400)
    return JsonResponse(combinedResult, status=200)


@csrf_exempt
@require_POST
def classify_intent_view(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message = (data.get("message") or data.get("text") or "").strip()
    if not message:
        return JsonResponse({"error": "Field 'message' is required."}, status=400)

    result = predict_intent(message)
    return JsonResponse(result, status=200)


@csrf_exempt
@require_POST
def classify_transaction_view(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    description = (
        data.get("description")
        or data.get("text")
        or data.get("message")
        or ""
    ).strip()
    if not description:
        return JsonResponse({"error": "Field 'description' is required."}, status=400)

    result = predict_transaction(description)
    return JsonResponse(result, status=200)


@csrf_exempt
@require_POST
def extract_transaction_item_view(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    text = (data.get("text") or data.get("message") or "").strip()
    if not text:
        return JsonResponse({"error": "Field 'text' is required."}, status=400)

    result = extract_expense(text)
    return JsonResponse(result, status=200)

