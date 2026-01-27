from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SurrogateModel:
    model_type: str = "random_forest"
    model: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=300, random_state=42)
        elif self.model_type == "mlp":
            self.model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
        elif self.model_type == "gbdt":
            self.model = GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SurrogateModel not trained yet.")
        return self.model.predict(X)
