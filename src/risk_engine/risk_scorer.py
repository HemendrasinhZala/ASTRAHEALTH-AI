
class AstronautHealthRiskScorer:
    """
    Experimental monitoring-priority engine.

    This is not a clinical risk prediction or diagnostic system.
    """

    def __init__(self):
        self.autoencoder_weight = 30
        self.isolation_forest_weight = 30
        self.high_reconstruction_weight = 20

    def calculate_score(
        self,
        autoencoder_status,
        isolation_forest_status,
        reconstruction_error,
        autoencoder_threshold
    ):

        score = 0
        explanations = []

        # 1. Autoencoder anomaly detection
        if autoencoder_status == "Anomaly Flag":
            score += self.autoencoder_weight
            explanations.append(
                "Autoencoder detected unusual reconstruction behavior."
            )

        # 2. Isolation Forest anomaly detection
        if isolation_forest_status == "Anomaly Flag":
            score += self.isolation_forest_weight
            explanations.append(
                "Isolation Forest detected an unusual data pattern."
            )

        # 3. High reconstruction error
        if reconstruction_error > 1.5 * autoencoder_threshold:
            score += self.high_reconstruction_weight
            explanations.append(
                "Reconstruction error is substantially above the model threshold."
            )

        # 4. Monitoring priority
        if score >= 60:
            priority = "Priority Review"
        elif score >= 30:
            priority = "Review Recommended"
        else:
            priority = "Routine Monitoring"

        if not explanations:
            explanations.append(
                "No anomaly indicators were triggered by the models."
            )

        return {
            "monitoring_priority_score": score,
            "monitoring_priority": priority,
            "explanations": explanations,
            "is_clinical_diagnosis": False
        }
