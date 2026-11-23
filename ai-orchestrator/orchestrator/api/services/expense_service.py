from api.services.category_service import load_model, run_prediction

expense_model = load_model("expense_model.joblib")


def predict_expense(description: str) -> dict:
    return run_prediction(expense_model, description)