
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocess import AstronautHealthPreprocessor


@pytest.fixture
def sample_data():

    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        rng.normal(size=(30, 15)),
        columns=[f"feature_{i}" for i in range(15)]
    )

    data["feature_0"] = data["feature_0"].astype(float)
    data.loc[0, "feature_0"] = np.nan

    for i in range(3):
        data[f"missing_{i}_MISSING"] = 0

    data.loc[0, "missing_0_MISSING"] = 1

    return data


def test_preprocessor_fit(sample_data):

    preprocessor = AstronautHealthPreprocessor()

    preprocessor.fit(sample_data)

    assert preprocessor.is_fitted is True


def test_output_shape(sample_data):

    preprocessor = AstronautHealthPreprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    assert transformed.shape == (30, 18)


def test_no_missing_values_after_imputation(sample_data):

    preprocessor = AstronautHealthPreprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    assert np.isfinite(transformed).all()


def test_transform_before_fit(sample_data):

    preprocessor = AstronautHealthPreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_data)
