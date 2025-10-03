from typing import List, Optional

import os
import joblib
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None


app = FastAPI(title="IMDB Sentiment API", version="1.0.0")


def load_model():
    model_path_candidates = [
        os.path.join("model", "logistic_regression_sentiment_model.joblib"),
        "logistic_regression_sentiment_model.joblib",
        os.path.join("model", "logreg_pipeline.joblib"),
        "logreg_pipeline.joblib",
        os.path.join("model", "logistic_regression_pipeline.joblib"),
        "logistic_regression_pipeline.joblib",
    ]
    for path in model_path_candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(
        "Model file 'logistic_regression_sentiment_model.joblib' not found under 'model/' or project root."
    )


try:
    model = load_model()
except Exception as exc:
    # Defer raising until first request so the app can start and report error via endpoint
    model = exc  # type: ignore


 


@app.get("/health")
def health() -> dict:
    if isinstance(model, Exception):
        return {"status": "error", "detail": str(model)}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if isinstance(model, Exception):
        raise HTTPException(status_code=500, detail=f"Model load error: {model}")

    inputs: List[str] = []
    if req.text is not None:
        inputs = [req.text]
    elif req.texts is not None:
        inputs = req.texts
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'.")

    try:
        # Unified pipeline (TF-IDF + LogisticRegression) expects raw texts
        preds = model.predict(inputs)  # type: ignore[attr-defined]
        # Try to compute positive class probabilities if available
        probs: Optional[List[float]] = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(inputs)  # type: ignore[attr-defined]
            # Assume positive sentiment is class 1 if binary
            if proba.shape[1] >= 2:
                probs = proba[:, 1].astype(float).tolist()
        return PredictResponse(predictions=[int(p) for p in preds.tolist()], probabilities=probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# Optional: form-data convenience endpoint for clients sending multipart/form-data
@app.post("/predict-form", response_model=PredictResponse)
def predict_form(text: str = Form(...)) -> PredictResponse:
    return predict(PredictRequest(text=text))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)