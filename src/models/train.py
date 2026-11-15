
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import AstronautHealthAutoencoder
from src.models.isolation_forest import AstronautHealthIsolationForest
from src.preprocessing.preprocess import AstronautHealthPreprocessor


class AstronautHealthMLPipeline:

    def __init__(
        self,
        input_dim=18,
        latent_dim=3,
        epochs=100,
        learning_rate=0.001,
        batch_size=4,
        random_state=42
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

        self.preprocessor = AstronautHealthPreprocessor()

        self.isolation_forest = (
            AstronautHealthIsolationForest(
                random_state=random_state
            )
        )

        self.autoencoder = AstronautHealthAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim
        ).to(self.device)

        self.threshold = None
        self.is_fitted = False

    def fit(self, X):

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Preprocess training data
        X_processed = self.preprocessor.fit_transform(X)

        # Train Isolation Forest
        self.isolation_forest.fit(X_processed)

        # Convert data to tensor
        X_tensor = torch.tensor(
            X_processed,
            dtype=torch.float32,
            device=self.device
        )

        dataset = TensorDataset(X_tensor, X_tensor)

        generator = torch.Generator().manual_seed(
            self.random_state
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator
        )

        # Configure Autoencoder training
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.MSELoss()

        self.autoencoder.train()

        for epoch in range(self.epochs):

            epoch_loss = 0.0

            for batch_data, batch_target in loader:

                optimizer.zero_grad()

                reconstructed = self.autoencoder(batch_data)

                loss = criterion(
                    reconstructed,
                    batch_target
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        # Calculate training reconstruction errors
        self.autoencoder.eval()

        with torch.no_grad():

            reconstructed = self.autoencoder(X_tensor)

            train_errors = torch.mean(
                (X_tensor - reconstructed) ** 2,
                dim=1
            ).cpu().numpy()

        # Preliminary training-derived threshold
        self.threshold = float(
            np.percentile(train_errors, 95)
        )

        self.is_fitted = True

        return self

    def predict(self, X):

        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before prediction."
            )

        # Apply fitted preprocessing
        X_processed = self.preprocessor.transform(X)

        # Isolation Forest predictions
        isolation_predictions = (
            self.isolation_forest.predict(X_processed)
        )

        isolation_scores = (
            self.isolation_forest.anomaly_scores(X_processed)
        )

        # Autoencoder predictions
        X_tensor = torch.tensor(
            X_processed,
            dtype=torch.float32,
            device=self.device
        )

        self.autoencoder.eval()

        with torch.no_grad():

            reconstruction_errors = (
                self.autoencoder.get_reconstruction_error(
                    X_tensor
                ).cpu().numpy()
            )

        autoencoder_predictions = np.where(
            reconstruction_errors > self.threshold,
            "Anomaly Flag",
            "No Anomaly Flag"
        )

        return {
            "isolation_forest_status": isolation_predictions,
            "isolation_forest_score": isolation_scores,
            "autoencoder_status": autoencoder_predictions,
            "reconstruction_error": reconstruction_errors,
            "autoencoder_threshold": self.threshold
        }
