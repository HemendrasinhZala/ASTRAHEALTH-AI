
import numpy as np

from sklearn.ensemble import IsolationForest


class AstronautHealthIsolationForest:
    """
    Isolation Forest model for astronaut health
    anomaly monitoring.
    """

    def __init__(
        self,
        n_estimators=100,
        contamination="auto",
        random_state=42
    ):

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

        self.is_fitted = False

    def fit(self, X):

        self.model.fit(X)

        self.is_fitted = True

        return self

    def predict(self, X):

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        predictions = self.model.predict(X)

        return np.where(
            predictions == -1,
            "Anomaly Flag",
            "No Anomaly Flag"
        )

    def anomaly_scores(self, X):

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before scoring."
            )

        return -self.model.score_samples(X)
