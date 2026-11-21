import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from api.services.intent_service import predict_intent


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