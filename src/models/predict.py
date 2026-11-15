
import json
import joblib
import torch

from pathlib import Path

from src.models.autoencoder import AstronautHealthAutoencoder
from src.risk_engine.risk_scorer import AstronautHealthRiskScorer


class AstronautHealthPredictor:

    def __init__(self):

        # Project root directory
        self.root_dir = Path(__file__).resolve().parents[2]

        # Select CPU or GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize model components
        self.autoencoder = None
        self.isolation_forest = None
        self.preprocessor = None
        self.threshold = None

        # Initialize risk scoring engine
        self.risk_scorer = AstronautHealthRiskScorer()

        self.is_loaded = False

    # --------------------------------------------------
    # LOAD SAVED MODELS
    # --------------------------------------------------

    def load_models(self):

        config_path = (
            self.root_dir / "artifacts/configs/model_config.json"
        )

        with open(config_path, "r") as file:
            config = json.load(file)

        # Load Autoencoder architecture
        self.autoencoder = AstronautHealthAutoencoder(
            input_dim=config["input_dimension"],
            latent_dim=config["latent_dimension"]
        ).to(self.device)

        # Load trained Autoencoder weights
        autoencoder_path = (
            self.root_dir / "artifacts/models/astronaut_autoencoder.pth"
        )

        state_dict = torch.load(
            autoencoder_path,
            map_location=self.device,
            weights_only=True
        )

        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.eval()

        # Load Isolation Forest
        self.isolation_forest = joblib.load(
            self.root_dir / "artifacts/models/isolation_forest.joblib"
        )

        # Load preprocessing pipeline
        self.preprocessor = joblib.load(
            self.root_dir / "artifacts/preprocessors/health_preprocessor.joblib"
        )

        # Load Autoencoder threshold
        self.threshold = config["autoencoder_threshold"]

        self.is_loaded = True

        print("ASTRAHEALTH AI models loaded successfully!")
        print("Device:", self.device)
        print("Autoencoder threshold:", self.threshold)

        return self

    # --------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------

    def predict(self, X):

        if not self.is_loaded:
            raise RuntimeError("Load models before prediction.")

        # 1. Preprocess input data
        X_processed = self.preprocessor.transform(X)

        # 2. Isolation Forest predictions
        isolation_status = self.isolation_forest.predict(
            X_processed
        )

        isolation_scores = self.isolation_forest.anomaly_scores(
            X_processed
        )

        # 3. Convert input into PyTorch tensor
        X_tensor = torch.tensor(
            X_processed,
            dtype=torch.float32,
            device=self.device
        )

        # 4. Autoencoder predictions
        self.autoencoder.eval()

        with torch.no_grad():

            reconstruction_errors = (
                self.autoencoder
                .get_reconstruction_error(X_tensor)
                .cpu()
                .numpy()
            )

        autoencoder_status = [
            "Anomaly Flag" if error > self.threshold
            else "No Anomaly Flag"
            for error in reconstruction_errors
        ]

        # 5. Calculate monitoring priorities
        risk_results = []

        for i in range(len(X_processed)):

            result = self.risk_scorer.calculate_score(
                autoencoder_status=autoencoder_status[i],
                isolation_forest_status=isolation_status[i],
                reconstruction_error=float(reconstruction_errors[i]),
                autoencoder_threshold=self.threshold
            )

            risk_results.append(result)

        # 6. Return complete prediction results
        return {
            "isolation_forest_status": isolation_status,
            "isolation_forest_score": isolation_scores,
            "autoencoder_status": autoencoder_status,
            "reconstruction_error": reconstruction_errors,
            "autoencoder_threshold": self.threshold,
            "risk_results": risk_results
        }
