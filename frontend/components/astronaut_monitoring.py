
import streamlit as st
import pandas as pd
import plotly.express as px

from src.models.inference_service import get_live_monitoring_data


# ==========================================================
# HEALTH MEASUREMENT CONFIGURATION
# ==========================================================

MEASUREMENT_COLUMNS = {
    "Hemoglobin": "hemoglobin_value_percent",
    "Hematocrit": "hematocrit_value_percent",
    "Red Blood Cell Count":
        "red_blood_cell_count_value_million_per_microliter",
    "White Blood Cell Count":
        "white_blood_cell_count_value_thousand_per_microliter",
    "Platelet Count":
        "platelet_count_value_thousand_per_microliter",
    "Mean Corpuscular Volume": "mcv_value_femtoliter",
    "Mean Corpuscular Hemoglobin": "mch_value_picogram",
    "Mean Corpuscular Hemoglobin Concentration":
        "mchc_value_grams_per_deciliter",
    "Red Cell Distribution Width": "rdw_value_percent",
    "Mean Platelet Volume": "mpv_value_femtoliter",
    "Neutrophils": "neutrophils_value_percent",
    "Lymphocytes": "lymphocytes_value_percent",
    "Monocytes": "monocytes_value_percent",
    "Eosinophils": "eosinophils_value_percent",
    "Basophils": "basophils_value_percent",
}


PRIORITY_COLORS = {
    "Routine Monitoring": "#22C55E",
    "Review Recommended": "#F59E0B",
    "Priority Review": "#EF4444",
}


# ==========================================================
# MAIN ASTRONAUT MONITORING MODULE
# ==========================================================

def render_astronaut_monitoring(monitoring_df, health_df):

    st.title("Astronaut Monitoring")

    st.caption(
        "Individual participant health monitoring using "
        "historical NASA OSDR Inspiration4 CBC observations "
        "and experimental machine learning inference."
    )

    st.info(
        "Research prototype: Model outputs are experimental "
        "anomaly indicators, not clinical diagnoses."
    )

    # ------------------------------------------------------
    # 1. DATA VALIDATION
    # ------------------------------------------------------

    required_monitoring = [
        "Sample Name",
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
    ]

    required_health = [
        "Sample Name",
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
    ]

    missing_monitoring = [
        column for column in required_monitoring
        if column not in monitoring_df.columns
    ]

    missing_health = [
        column for column in required_health
        if column not in health_df.columns
    ]

    if missing_monitoring or missing_health:
        st.error(
            "Required dataset columns are missing. "
            f"Monitoring: {missing_monitoring}; "
            f"Health: {missing_health}"
        )
        return

    if monitoring_df.empty or health_df.empty:
        st.warning("No astronaut observations are available.")
        return

    # ------------------------------------------------------
    # 2. LIVE ML INFERENCE
    # ------------------------------------------------------

    st.subheader("Live Model Analysis")

    try:
        metadata = monitoring_df[
            required_monitoring
        ].copy()

        live_df = get_live_monitoring_data(metadata)

        live_df = live_df.rename(columns={
            "monitoring_priority_score":
                "MONITORING_PRIORITY_SCORE",
            "monitoring_priority":
                "MONITORING_PRIORITY",
            "explanations":
                "EXPLANATIONS",
        })

        st.success(
            f"ML inference successful: "
            f"{len(live_df)} observations processed."
        )

    except Exception as error:
        st.error(f"Live model inference failed: {error}")
        return

    if live_df.empty:
        st.warning("No model predictions are available.")
        return

    # ------------------------------------------------------
    # 3. PREPARE DATA
    # ------------------------------------------------------

    live_df["MISSION_DAY"] = pd.to_numeric(
        live_df["MISSION_DAY"],
        errors="coerce",
    )

    live_df["MONITORING_PRIORITY_SCORE"] = pd.to_numeric(
        live_df["MONITORING_PRIORITY_SCORE"],
        errors="coerce",
    )

    live_df["RECONSTRUCTION_ERROR"] = pd.to_numeric(
        live_df["RECONSTRUCTION_ERROR"],
        errors="coerce",
    )

    health = health_df.copy()

    health["MISSION_DAY"] = pd.to_numeric(
        health["MISSION_DAY"],
        errors="coerce",
    )

    # ------------------------------------------------------
    # 4. PARTICIPANT SELECTION
    # ------------------------------------------------------

    st.divider()

    st.subheader("Select Astronaut")

    astronaut_ids = sorted(
        live_df["SUBJECT_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not astronaut_ids:
        st.warning("No astronaut participants found.")
        return

    selected_astronaut = st.selectbox(
        "Astronaut Participant",
        options=astronaut_ids,
        key="astronaut_monitoring_selection",
    )

    astronaut_monitoring = live_df[
        live_df["SUBJECT_ID"].astype(str)
        == selected_astronaut
    ].copy()

    astronaut_health = health[
        health["SUBJECT_ID"].astype(str)
        == selected_astronaut
    ].copy()

    astronaut_monitoring = astronaut_monitoring.sort_values(
        "MISSION_DAY"
    ).reset_index(drop=True)

    astronaut_health = astronaut_health.sort_values(
        "MISSION_DAY"
    ).reset_index(drop=True)

    if astronaut_monitoring.empty or astronaut_health.empty:
        st.warning(
            "No matching monitoring or health records found "
            "for this participant."
        )
        return

    # ------------------------------------------------------
    # 5. PARTICIPANT PROFILE
    # ------------------------------------------------------

    latest = astronaut_monitoring.iloc[-1]

    total_observations = len(astronaut_monitoring)

    review_count = int(
        astronaut_monitoring["MONITORING_PRIORITY"].isin([
            "Review Recommended",
            "Priority Review",
        ]).sum()
    )

    routine_count = int(
        (
            astronaut_monitoring["MONITORING_PRIORITY"]
            == "Routine Monitoring"
        ).sum()
    )

    st.divider()

    st.subheader(
        f"Participant Profile: {selected_astronaut}"
    )

    profile_columns = st.columns(4)

    profile_columns[0].metric(
        "Observations",
        total_observations,
    )

    profile_columns[1].metric(
        "Latest Mission Day",
        int(latest["MISSION_DAY"]),
    )

    profile_columns[2].metric(
        "Routine Observations",
        routine_count,
    )

    profile_columns[3].metric(
        "Review Flags",
        review_count,
    )

    # ------------------------------------------------------
    # 6. LATEST MODEL RESULT
    # ------------------------------------------------------

    st.divider()

    st.subheader("Latest ML Monitoring Result")

    priority = str(latest["MONITORING_PRIORITY"])

    result_columns = st.columns(3)

    result_columns[0].metric(
        "Monitoring Priority",
        priority,
    )

    result_columns[1].metric(
        "Monitoring Score",
        f"{latest['MONITORING_PRIORITY_SCORE']:.0f}/80",
    )

    result_columns[2].metric(
        "Reconstruction Error",
        f"{latest['RECONSTRUCTION_ERROR']:.4f}",
    )

    if priority == "Priority Review":
        st.error("Priority human review recommended.")

    elif priority == "Review Recommended":
        st.warning("Additional observation review recommended.")

    else:
        st.success("Routine research monitoring category.")

    st.markdown("**Actual Model Predictions**")

    model_columns = st.columns(2)

    with model_columns[0]:
        st.markdown("**Isolation Forest**")
        st.write(latest["ISOLATION_FOREST_STATUS"])

        st.metric(
            "Anomaly Score",
            f"{latest['ISOLATION_FOREST_SCORE']:.4f}",
        )

    with model_columns[1]:
        st.markdown("**PyTorch Autoencoder**")
        st.write(latest["AUTOENCODER_STATUS"])

        st.metric(
            "Reconstruction Error",
            f"{latest['RECONSTRUCTION_ERROR']:.4f}",
        )

    st.markdown("**Model Explanation**")

    explanations = latest["EXPLANATIONS"]

    if isinstance(explanations, list):
        for explanation in explanations:
            st.write(f"- {explanation}")
    else:
        st.write(str(explanations))

    st.caption(
        "These outputs are experimental model-generated "
        "monitoring indicators, not clinical assessments."
    )

    # ------------------------------------------------------
    # 7. HISTORICAL HEALTH MEASUREMENTS
    # ------------------------------------------------------

    st.divider()

    st.subheader("Historical Health Measurements")

    available_measurements = {
        name: column
        for name, column in MEASUREMENT_COLUMNS.items()
        if column in astronaut_health.columns
    }

    if not available_measurements:
        st.warning(
            "No supported health measurements found."
        )
        return

    selected_measurement = st.selectbox(
        "Choose Health Measurement",
        options=list(available_measurements.keys()),
        key="astronaut_health_measurement",
    )

    measurement_column = available_measurements[
        selected_measurement
    ]

    chart_data = astronaut_health.copy()

    chart_data[measurement_column] = pd.to_numeric(
        chart_data[measurement_column],
        errors="coerce",
    )

    chart_data = chart_data.dropna(
        subset=["MISSION_DAY", measurement_column]
    )

    if chart_data.empty:

        st.info(
            "No valid values are available for this measurement."
        )

    else:

        fig = px.line(
            chart_data,
            x="MISSION_DAY",
            y=measurement_column,
            markers=True,
            hover_data=["TIMEPOINT"],
            title=(
                f"{selected_measurement} — "
                f"{selected_astronaut}"
            ),
        )

        fig.update_traces(
            line=dict(
                color="#38BDF8",
                width=3,
            ),
            marker=dict(
                color="#22D3EE",
                size=9,
            ),
        )

        fig.update_layout(
            paper_bgcolor="#101E31",
            plot_bgcolor="#101E31",
            font=dict(color="#E6F0FC"),
            xaxis_title="Mission Day",
            yaxis_title=selected_measurement,
            xaxis=dict(gridcolor="#223752"),
            yaxis=dict(gridcolor="#223752"),
            height=430,
            margin=dict(
                l=25,
                r=25,
                t=55,
                b=35,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="astronaut_profile_health_chart",
        )

        st.caption(
            f"{len(chart_data)} valid measurements displayed."
        )

    # ------------------------------------------------------
    # 8. MODEL MONITORING HISTORY
    # ------------------------------------------------------

    st.divider()

    st.subheader("Live Model Monitoring History")

    history_columns = [
        "TIMEPOINT",
        "MISSION_DAY",
        "ISOLATION_FOREST_STATUS",
        "AUTOENCODER_STATUS",
        "RECONSTRUCTION_ERROR",
        "MONITORING_PRIORITY_SCORE",
        "MONITORING_PRIORITY",
    ]

    history_df = astronaut_monitoring[
        history_columns
    ].copy()

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------
    # 9. OBSERVATION DETAILS
    # ------------------------------------------------------

    st.divider()

    st.subheader("Observation Details")

    observation_options = astronaut_monitoring[
        "Sample Name"
    ].astype(str).tolist()

    selected_sample = st.selectbox(
        "Select Observation",
        options=observation_options,
        key="astronaut_observation_selection",
    )

    observation = astronaut_monitoring[
        astronaut_monitoring["Sample Name"].astype(str)
        == selected_sample
    ].iloc[0]

    detail_columns = st.columns(2)

    with detail_columns[0]:

        st.write(
            f"**Sample:** {observation['Sample Name']}"
        )

        st.write(
            f"**Timepoint:** {observation['TIMEPOINT']}"
        )

        st.write(
            f"**Mission Day:** {observation['MISSION_DAY']}"
        )

        st.write(
            f"**Priority:** {observation['MONITORING_PRIORITY']}"
        )

    with detail_columns[1]:

        st.write(
            f"**Priority Score:** "
            f"{observation['MONITORING_PRIORITY_SCORE']}"
        )

        st.write(
            f"**Isolation Forest:** "
            f"{observation['ISOLATION_FOREST_STATUS']}"
        )

        st.write(
            f"**Autoencoder:** "
            f"{observation['AUTOENCODER_STATUS']}"
        )

        st.write(
            f"**Reconstruction Error:** "
            f"{observation['RECONSTRUCTION_ERROR']:.4f}"
        )

    st.info(str(observation["EXPLANATIONS"]))

    # ------------------------------------------------------
    # 10. EXPORT PARTICIPANT HISTORY
    # ------------------------------------------------------

    st.divider()

    st.subheader("Export Participant Records")

    csv_data = astronaut_monitoring.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Participant Monitoring CSV",
        data=csv_data,
        file_name=(
            f"{selected_astronaut}_live_monitoring_history.csv"
        ),
        mime="text/csv",
        key="astronaut_monitoring_download",
        use_container_width=True,
    )

    st.caption(
        "ASTRAHEALTH AI | Historical research observations "
        "and experimental ML outputs | Not a clinical system"
    )
