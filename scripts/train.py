from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train, params: dict):
    model_type = params.get("type", "logistic_regression")
    random_state = params.get("random_state", 42)

    if model_type == "logistic_regression":
        # Income and loan amounts are in dollars, so scale before the regression or it will not converge
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=params.get("max_iter", 500), random_state=random_state),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def save_model(model, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
