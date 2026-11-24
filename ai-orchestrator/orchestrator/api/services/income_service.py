from api.services.category_service import load_model, run_prediction

income_model = load_model("income_model.joblib")


def predict_income(description: str) -> dict:
    return run_prediction(income_model, description)