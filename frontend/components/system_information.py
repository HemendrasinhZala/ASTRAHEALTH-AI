
import streamlit as st
import pandas as pd
import json
from pathlib import Path


def render_system_information(monitoring_df, health_df):

    st.title("System Information")

    st.caption(
        "ASTRAHEALTH AI | Technical architecture, "
        "datasets, models and system configuration."
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    st.info(
        "ASTRAHEALTH AI is an experimental research prototype "
        "using historical NASA OSDR data. It is not a clinical system."
    )

    # --------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------

    st.subheader("Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Astronaut Participants", monitoring_df["SUBJECT_ID"].nunique())
    c2.metric("Historical Observations", len(monitoring_df))
    c3.metric("Health Data Features", health_df.shape[1])

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------

    st.subheader("Technology Stack")

    stack = {
        "Programming Language": "Python 3.13",
        "Frontend": "Streamlit",
        "Visualization": "Plotly",
        "Deep Learning": "PyTorch",
        "Machine Learning": "Scikit-learn",
        "Data Processing": "Pandas, NumPy",
        "Dataset": "NASA OSDR – OSD-569",
    }

    st.dataframe(
        pd.DataFrame(
            list(stack.items()),
            columns=["Component", "Technology"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------

    st.subheader("Dataset Information")

    st.markdown("""
    **Dataset:** NASA Open Science Data Repository (OSDR)

    **Study:** OSD-569 – Inspiration4 Whole Blood Measurements

    **Participants:** C001, C002, C003 and C004

    **Observation timepoints:** L-92, L-44, L-3, R+1,
    R+45, R+82 and R+194.

    **Processed observations:** 27

    **Source:** [NASA OSDR – OSD-569](https://osdr.nasa.gov/bio/repo/data/studies/OSD-569/)
    """)

    st.warning(
        "The dataset contains historical research observations, "
        "not real-time astronaut telemetry."
    )

    st.divider()

    # --------------------------------------------------
    # MACHINE LEARNING ARCHITECTURE
    # --------------------------------------------------

    st.subheader("AI Model Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PyTorch Autoencoder")

        st.write("- Model type: Neural network Autoencoder")
        st.write("- Input dimensions: 18")
        st.write("- Latent dimensions: 3")
        st.write("- Training epochs: 100")
        st.write("- Optimizer: Adam")
        st.write("- Loss: Mean Squared Error")
        st.write("- Threshold: 95th percentile of training reconstruction error")

    with col2:
        st.markdown("### Isolation Forest")

        st.write("- Model type: Unsupervised anomaly detection")
        st.write("- Estimators: 100")
        st.write("- Library: Scikit-learn")
        st.write("- Input: Processed health features")
        st.write("- Output: Experimental anomaly flags")

    st.divider()

    # --------------------------------------------------
    # MODEL ARTIFACTS
    # --------------------------------------------------

    st.subheader("Saved Model Artifacts")

    artifacts = [
        (
            "PyTorch Autoencoder",
            "artifacts/models/astronaut_autoencoder.pth",
        ),
        (
            "Isolation Forest",
            "artifacts/models/isolation_forest.joblib",
        ),
        (
            "Health Preprocessor",
            "artifacts/preprocessors/health_preprocessor.joblib",
        ),
        (
            "Model Configuration",
            "artifacts/configs/model_config.json",
        ),
    ]

    for name, relative_path in artifacts:

        full_path = PROJECT_ROOT / relative_path

        exists = full_path.exists()

        col1, col2 = st.columns([3, 1])

        col1.write(f"**{name}**")
        col1.caption(relative_path)

        if exists:
            col2.success("AVAILABLE")
        else:
            col2.error("MISSING")

    st.divider()

    # --------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------

    st.subheader("Model Configuration")

    config_path = (
        PROJECT_ROOT
        / "artifacts"
        / "configs"
        / "model_config.json"
    )

    if config_path.exists():

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)

            st.json(config)

        except Exception as error:
            st.warning(f"Could not read configuration: {error}")

    else:
        st.warning("Model configuration file was not found.")

    st.divider()

    # --------------------------------------------------
    # PROJECT ARCHITECTURE
    # --------------------------------------------------

    st.subheader("System Architecture")

    st.code("""
NASA OSDR Historical Dataset
             |
             v
     Data Preprocessing
             |
             v
       Feature Engineering
             |
       +-----+------+
       |            |
       v            v
 Isolation      PyTorch
  Forest       Autoencoder
       |            |
       +-----+------+
             |
             v
      Risk Scoring Engine
             |
             v
    Decision Support System
             |
             v
       Agentic Workflow
             |
             v
      Streamlit Dashboard
    """, language="text")

    st.divider()

    # --------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------

    st.subheader("Research Limitations")

    limitations = [
        "Small dataset containing only 27 observations.",
        "Only four astronaut participants.",
        "No clinically validated anomaly labels.",
        "Model outputs are experimental and not diagnostic.",
        "Historical records do not represent live telemetry.",
        "Monitoring thresholds are engineering rules, not medical thresholds.",
        "Model validation does not establish clinical accuracy.",
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.divider()

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.markdown("""
    **ASTRAHEALTH AI**

    Autonomous Astronaut Health Monitoring and Decision Support System

    NASA Space Apps Challenge 2026

    *Experimental research and demonstration prototype.*
    """)

