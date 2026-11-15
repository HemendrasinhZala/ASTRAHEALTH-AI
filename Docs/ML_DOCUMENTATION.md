# ASTRAHEALTH AI
## Machine Learning Technical Documentation

**Project:** ASTRAHEALTH AI – Autonomous Astronaut Health Monitoring and Decision Support System  
**Challenge:** NASA Space Apps Challenge 2026 – Health Monitoring Software for Astronauts on Space Missions  
**ML Framework:** PyTorch, Scikit-learn  
**Programming Language:** Python 3.13.5  
**Execution Device:** CPU  
**Document Version:** 1.0

---

# 1. Project Overview

ASTRAHEALTH AI is an experimental astronaut health monitoring and decision-support system designed to analyze physiological measurements and identify unusual patterns using machine learning.

The system combines two unsupervised machine learning algorithms:

1. Isolation Forest
2. Deep Learning Autoencoder

Their outputs are integrated into an explainable monitoring-priority engine.

The objective is to provide an experimental monitoring interface that can help identify unusual patterns in astronaut health measurements.

The system is not intended to diagnose medical conditions or replace professional medical assessment.

# 2. Dataset Description

## 2.1 Dataset Source

**Dataset:** NASA Open Science Data Repository (OSDR)  
**Study ID:** OSD-569  
**Study:** Inspiration4 Whole Blood Measurements

Dataset URL:

https://osdr.nasa.gov/bio/repo/data/studies/OSD-569/

## 2.2 Dataset Information

| Attribute | Description |
|---|---|
| Participants | 4 |
| Subject IDs | C001, C002, C003, C004 |
| Original timepoints | 7 |
| Original CBC samples | 28 |
| Usable ML samples | 27 |
| Selected measurement features | 15 |
| Missing-value indicator features | 3 |
| Final model input dimensions | 18 |

The seven timepoints are:

* L-92
* L-44
* L-3
* R+1
* R+45
* R+82
* R+194

## 2.3 Data Quality Assessment

The original dataset was inspected for:

* Missing values
* Duplicate observations
* Measurement inconsistencies
* Differential blood count consistency
* Out-of-range measurements

One sample, C003 at L-92, showed serious cross-consistency issues in differential blood count measurements.

This sample was excluded from the initial ML dataset rather than correcting uncertain source values.

As a result, 27 observations were retained.

# 3. Feature Engineering

## 3.1 Selected Measurement Features

The following 15 features were selected:

1. Basophils percentage
2. Eosinophils percentage
3. Lymphocytes percentage
4. Monocytes percentage
5. Neutrophils percentage
6. Hematocrit
7. Hemoglobin
8. Mean Corpuscular Hemoglobin (MCH)
9. Mean Corpuscular Hemoglobin Concentration (MCHC)
10. Mean Corpuscular Volume (MCV)
11. Red Cell Distribution Width (RDW)
12. Red Blood Cell Count
13. Mean Platelet Volume (MPV)
14. Platelet Count
15. White Blood Cell Count

Five absolute differential count features were excluded due to the identified source-data consistency issue.

## 3.2 Missing-Value Indicators

Three missing-value indicator columns were added.

These indicators preserve information about missing measurements after numerical imputation.

Final feature matrix:

**27 samples × 18 features**

# 4. Data Preprocessing

The preprocessing pipeline uses Scikit-learn.

### Numerical preprocessing

1. Median imputation for missing numerical measurements.
2. StandardScaler for numerical feature normalization.
3. Missing-value indicator columns passed through without scaling.

The preprocessing transformation is implemented in:

`src/preprocessing/preprocess.py`

The preprocessor is fitted on training data during cross-validation and then applied to the corresponding held-out participant.

This prevents information leakage during the participant-level validation experiments.

# 5. Machine Learning Architecture

ASTRAHEALTH AI uses two complementary unsupervised anomaly-detection models.

## 5.1 Isolation Forest

Isolation Forest identifies unusual observations by isolating data points through random feature selection and splitting.

Implementation:

`src/models/isolation_forest.py`

Configuration:

| Parameter | Value |
|---|---|
| Algorithm | Isolation Forest |
| Estimators | 100 |
| Contamination | auto |
| Random State | 42 |

Output:

* Anomaly Flag
* No Anomaly Flag
* Anomaly score

The model provides an independent anomaly-detection signal.

## 5.2 PyTorch Autoencoder

The Autoencoder learns to reconstruct its input features.

Unusual observations may produce larger reconstruction errors because they differ from the patterns learned during training.

Implementation:

`src/models/autoencoder.py`

### Architecture

Input dimension: 18

Encoder:

* Linear: 18 → 12
* ReLU
* Linear: 12 → 6
* ReLU
* Linear: 6 → 3

Decoder:

* Linear: 3 → 6
* ReLU
* Linear: 6 → 12
* ReLU
* Linear: 12 → 18

Latent dimension: 3

Total trainable parameters: 669

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Loss Function | Mean Squared Error |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Batch Size | 4 |
| Random State | 42 |
| Device | CPU |

### Reconstruction Error

The model calculates the mean squared reconstruction error for each observation.

A higher reconstruction error indicates that the model had more difficulty reconstructing the sample.

# 6. Model Validation

## 6.1 Validation Strategy

Leave-One-Group-Out cross-validation was used.

The grouping variable was SUBJECT_ID.

Four validation folds were created.

| Fold | Training Participants | Held-Out Participant |
|---|---|---|
| 1 | C002, C003, C004 | C001 |
| 2 | C001, C003, C004 | C002 |
| 3 | C001, C002, C004 | C003 |
| 4 | C001, C002, C003 | C004 |

This approach evaluates whether models trained on three participants can generalize to a fourth participant.

## 6.2 Isolation Forest Results

The original four-fold experiment produced:

| Participant | Anomaly Flags | No Anomaly Flags |
|---|---:|---:|
| C001 | 3 | 4 |
| C002 | 7 | 0 |
| C003 | 4 | 2 |
| C004 | 7 | 0 |
| **Total** | **21** | **6** |

## 6.3 Autoencoder Results

The held-out reconstruction error analysis produced:

| Participant | Mean Reconstruction Error |
|---|---:|
| C001 | 0.741030 |
| C002 | 0.971697 |
| C003 | 2.616249 |
| C004 | 2.697129 |

Overall reconstruction-error statistics:

| Metric | Value |
|---|---:|
| Mean | 1.7247 |
| Median | 1.5682 |
| Standard Deviation | 1.0505 |
| Minimum | 0.3520 |
| Maximum | 3.8349 |

These statistics describe reconstruction behavior across the held-out observations.

Higher errors for C003 and C004 indicate differences in reconstruction performance across participants. They do not establish medical abnormalities.

## 6.4 Autoencoder Threshold Evaluation

An experimental threshold was calculated using the 95th percentile of training reconstruction errors within each fold.

| Fold | Threshold | Held-Out Samples | Flagged |
|---|---:|---:|---:|
| 1 | 0.502185 | 7 | 4 |
| 2 | 0.470987 | 7 | 7 |
| 3 | 0.550865 | 6 | 6 |
| 4 | 0.932333 | 7 | 7 |
| **Total** | | **27** | **24** |

Overall:

* 24 anomaly flags
* 3 samples without anomaly flags
* 88.89% experimental flag rate

The high flag rate suggests that the learned training distributions do not consistently generalize to held-out participants.

The threshold is experimental and has not been clinically validated.

## 6.5 Model Comparison

Isolation Forest and Autoencoder flags were compared on the same held-out observations.

| Comparison | Result |
|---|---:|
| Total observations | 27 |
| Model agreement | 24 |
| Model disagreement | 3 |
| Agreement percentage | 88.9% |

This is agreement between two anomaly-detection models, not classification accuracy.

# 7. Feature-Level Explanations

The Autoencoder reconstruction errors were analyzed at feature level.

Squared reconstruction errors were calculated for each input feature.

The analysis identified measurements that contributed more strongly to reconstruction error.

Features frequently appearing among the top five contributors included:

* Red Blood Cell Count
* MCHC
* Hematocrit
* Platelet Count
* MPV
* RDW

These are model reconstruction contributions, not clinical risk rankings.

Feature-level explanation files:

* `autoencoder_feature_errors.csv`
* `autoencoder_top_features.csv`
* `autoencoder_feature_summary.csv`

# 8. Model Artifact Management

The trained models and preprocessing components are saved for reuse.

| Artifact | Location |
|---|---|
| Autoencoder weights | `artifacts/models/astronaut_autoencoder.pth` |
| Isolation Forest | `artifacts/models/isolation_forest.joblib` |
| Preprocessor | `artifacts/preprocessors/health_preprocessor.joblib` |
| Model configuration | `artifacts/configs/model_config.json` |

The configuration stores:

* Input dimension
* Latent dimension
* Training epochs
* Learning rate
* Batch size
* Threshold percentile
* Autoencoder threshold
* Training sample count
* Random state
* Execution device

# 9. Integrated Prediction Pipeline

The inference module is implemented in:

`src/models/predict.py`

The `AstronautHealthPredictor` class performs the following operations:

1. Loads model configuration.
2. Loads trained Autoencoder weights.
3. Loads Isolation Forest.
4. Loads the preprocessing pipeline.
5. Transforms incoming data.
6. Generates Isolation Forest predictions.
7. Calculates Autoencoder reconstruction errors.
8. Applies the experimental reconstruction threshold.
9. Calculates monitoring-priority scores.
10. Returns explanations and prediction outputs.

The pipeline was successfully tested on 27 observations.

The packaged-model integration test uses the same observations that were used for final model training. Its predictions are therefore not independent performance measurements.

# 10. Experimental Monitoring-Priority Engine

Implementation:

`src/risk_engine/risk_scorer.py`

The monitoring-priority engine combines model outputs into an experimental score.

## 10.1 Scoring Rules

| Condition | Points |
|---|---:|
| Autoencoder anomaly flag | +30 |
| Isolation Forest anomaly flag | +30 |
| Reconstruction error exceeds 1.5 × threshold | +20 |
| Maximum possible score | 80 |

## 10.2 Monitoring Priority Categories

| Score | Category |
|---:|---|
| 0–20 | Routine Monitoring |
| 30–50 | Review Recommended |
| 60–80 | Priority Review |

The scoring system also generates textual explanations describing the triggered conditions.

These weights and category boundaries are initial engineering choices. They are not derived from validated clinical risk models.

The score must not be interpreted as a probability of disease, medical severity or a clinical diagnosis.

# 11. Software Testing

Pytest was used for unit and integration testing.

Five test modules were created.

| Test File | Number of Tests |
|---|---:|
| `test_risk_scorer.py` | 4 |
| `test_autoencoder.py` | 4 |
| `test_isolation_forest.py` | 4 |
| `test_preprocessing.py` | 4 |
| `test_predictor.py` | 5 |
| **Total** | **21** |

All 26 tests passed.

### Testing Coverage

The tests verify:

* Monitoring-priority calculations
* Explanation output
* Autoencoder output dimensions
* Reconstruction-error shape
* Non-negative reconstruction errors
* Model parameter count
* Isolation Forest fitting and prediction
* Anomaly score generation
* Prediction before model fitting
* Missing-value imputation
* Preprocessing output dimensions
* Predictor model loading
* Prediction output structure
* Monitoring score validity
* Finite reconstruction errors

These tests verify software behavior and integration. They do not establish clinical accuracy or real-world medical safety.

# 12. Project Directory Structure

```text
ASTRAHEALTH_AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│   └── 03_ml_model_development.ipynb
│
├── src/
│   ├── agents/
│   ├── data_collection/
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── isolation_forest.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   └── risk_engine/
│       └── risk_scorer.py
│
├── artifacts/
│   ├── configs/
│   ├── models/
│   ├── preprocessors/
│   └── thresholds/
│
├── tests/
│   ├── test_autoencoder.py
│   ├── test_isolation_forest.py
│   ├── test_predictor.py
│   ├── test_preprocessing.py
│   └── test_risk_scorer.py
│
└── docs/
    └── ML_DOCUMENTATION.md
```

# 13. Limitations and Future Improvements

## 13.1 Dataset Limitations

* Only four participants are included.
* Only 27 usable observations are available.
* The dataset is too small to establish robust population-level conclusions.
* Individual participant differences can strongly influence model behavior.

## 13.2 Model Limitations

* No verified clinical anomaly labels are available for conventional supervised classification evaluation.
* High anomaly-flag rates require further investigation.
* The experimental thresholds may be unstable due to small training sets.
* Model agreement does not establish model correctness.
* The monitoring-priority weights are not clinically validated.

## 13.3 Future Improvements

1. Evaluate additional astronaut health datasets.
2. Increase the number of participants and observations.
3. Investigate participant-specific and temporal modeling approaches.
4. Conduct sensitivity analysis for anomaly thresholds.
5. Compare additional anomaly-detection algorithms.
6. Improve feature-level explanations.
7. Validate alert rules with qualified medical experts.
8. Evaluate robustness under distribution shifts.
9. Establish clinically meaningful validation criteria before any real-world medical use.

# 14. Conclusion

ASTRAHEALTH AI demonstrates an experimental machine learning architecture combining a PyTorch Autoencoder, Isolation Forest, preprocessing pipeline and explainable monitoring-priority engine.

The system successfully processes NASA Inspiration4 blood measurement data, performs participant-level exploratory validation, saves trained model artifacts, generates integrated predictions and passes 21 software unit and integration tests.

The project provides a foundation for an astronaut health monitoring dashboard and future research into explainable AI-based health monitoring.

**Important:** ASTRAHEALTH AI is a research and demonstration prototype. Its outputs are experimental monitoring indicators, not medical diagnoses or clinically validated risk assessments.
