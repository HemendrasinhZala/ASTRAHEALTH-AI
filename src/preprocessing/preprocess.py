
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AstronautHealthPreprocessor:
    """
    Preprocessing pipeline for astronaut CBC data.

    Numeric features:
        Median imputation + StandardScaler

    Missing indicators:
        Passed through without scaling.
    """

    def __init__(self):

        self.numeric_columns = None
        self.missing_flag_columns = None
        self.preprocessor = None
        self.is_fitted = False

    def fit(self, X):

        self.missing_flag_columns = [
            col for col in X.columns
            if col.endswith("_MISSING")
        ]

        self.numeric_columns = [
            col for col in X.columns
            if col not in self.missing_flag_columns
        ]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns
                ),
                (
                    "missing_flags",
                    "passthrough",
                    self.missing_flag_columns
                )
            ],
            remainder="drop"
        )

        self.preprocessor.fit(X)

        self.is_fitted = True

        return self

    def transform(self, X):

        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted first."
            )

        expected_columns = (
            self.numeric_columns + self.missing_flag_columns
        )

        if set(X.columns) != set(expected_columns):
            raise ValueError(
                "Input features do not match fitted features."
            )

        X = X[expected_columns]

        return self.preprocessor.transform(X)

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
