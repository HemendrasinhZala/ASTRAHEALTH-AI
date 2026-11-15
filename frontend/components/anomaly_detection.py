
import streamlit as st
import pandas as pd
import plotly.express as px

from src.models.inference_service import get_live_monitoring_data


PRIORITY_COLORS = {
    "Routine Monitoring": "#00C875",
    "Review Recommended": "#FFB020",
    "Priority Review": "#FF4B4B",
}


def render_anomaly_detection(monitoring_df):

    st.title("AI Anomaly Detection")

    st.caption(
        "Actual model inference using PyTorch Autoencoder "
        "and Isolation Forest."
    )

    st.info(
        "Experimental anomaly indicators based on historical "
        "NASA OSDR data. Not clinical diagnoses."
    )

    # --------------------------------------------------
    # LIVE MODEL INFERENCE
    # --------------------------------------------------

    st.subheader("Live Model Analysis")

    try:
        metadata = monitoring_df[
            [
                "Sample Name",
                "SUBJECT_ID",
                "TIMEPOINT",
                "MISSION_DAY",
            ]
        ].copy()

        df = get_live_monitoring_data(metadata)

        df = df.rename(columns={
            "monitoring_priority_score":
                "MONITORING_PRIORITY_SCORE",
            "monitoring_priority":
                "MONITORING_PRIORITY",
            "explanations":
                "EXPLANATIONS",
        })

        st.success(
            f"Successfully analyzed {len(df)} observations "
            "using trained ML models."
        )

    except Exception as error:
        st.error(f"Live inference failed: {error}")
        return

    # --------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------

    df["RECONSTRUCTION_ERROR"] = pd.to_numeric(
        df["RECONSTRUCTION_ERROR"], errors="coerce"
    )

    df["MONITORING_PRIORITY_SCORE"] = pd.to_numeric(
        df["MONITORING_PRIORITY_SCORE"], errors="coerce"
    )

    df["MISSION_DAY"] = pd.to_numeric(
        df["MISSION_DAY"], errors="coerce"
    )

    total = len(df)

    # Exact matching prevents counting "No Anomaly Flag".
    if_total = (
        df["ISOLATION_FOREST_STATUS"]
        .astype(str).str.strip().eq("Anomaly Flag")
        .sum()
    )

    ae_total = (
        df["AUTOENCODER_STATUS"]
        .astype(str).str.strip().eq("Anomaly Flag")
        .sum()
    )

    priority_total = (
        df["MONITORING_PRIORITY"]
        == "Priority Review"
    ).sum()

    review_total = (
        df["MONITORING_PRIORITY"]
        == "Review Recommended"
    ).sum()

    # --------------------------------------------------
    # KPI SUMMARY
    # --------------------------------------------------

    st.subheader("Detection Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Observations", total)
    c2.metric("Isolation Forest Flags", int(if_total))
    c3.metric("Autoencoder Flags", int(ae_total))
    c4.metric("Priority Reviews", int(priority_total))

    st.divider()

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    st.subheader("Model Detection Comparison")

    comparison = pd.DataFrame({
        "Model": ["Isolation Forest", "Autoencoder"],
        "Flagged Observations": [
            int(if_total),
            int(ae_total),
        ],
        "Not Flagged": [
            total - int(if_total),
            total - int(ae_total),
        ],
    })

    comparison_long = comparison.melt(
        id_vars="Model",
        var_name="Detection",
        value_name="Observations",
    )

    fig = px.bar(
        comparison_long,
        x="Model",
        y="Observations",
        color="Detection",
        barmode="group",
        text="Observations",
        title="Anomaly Flags Across Both Models",
        color_discrete_sequence=["#FF4B4B", "#00C875"],
    )

    fig.update_layout(
        paper_bgcolor="#0B1422",
        plot_bgcolor="#0B1422",
        font_color="#EAF2FF",
        legend_title_text="",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Monitoring Priority Distribution")

    priority_order = [
        "Routine Monitoring",
        "Review Recommended",
        "Priority Review",
    ]

    priority_counts = (
        df["MONITORING_PRIORITY"]
        .value_counts()
        .reindex(priority_order, fill_value=0)
        .rename_axis("Priority")
        .reset_index(name="Observations")
    )

    fig_priority = px.pie(
        priority_counts,
        names="Priority",
        values="Observations",
        hole=0.55,
        color="Priority",
        color_discrete_map=PRIORITY_COLORS,
    )

    fig_priority.update_layout(
        paper_bgcolor="#0B1422",
        font_color="#EAF2FF",
        legend_title_text="",
    )

    st.plotly_chart(
        fig_priority,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # RECONSTRUCTION ERROR
    # --------------------------------------------------

    st.subheader("Autoencoder Reconstruction Analysis")

    error_df = df.dropna(
        subset=["RECONSTRUCTION_ERROR"]
    ).copy()

    if not error_df.empty:

        fig_error = px.scatter(
            error_df,
            x="MISSION_DAY",
            y="RECONSTRUCTION_ERROR",
            color="MONITORING_PRIORITY",
            hover_data=[
                "SUBJECT_ID",
                "TIMEPOINT",
                "AUTOENCODER_STATUS",
            ],
            color_discrete_map=PRIORITY_COLORS,
            title="Reconstruction Error Across Mission Timeline",
        )

        fig_error.update_layout(
            paper_bgcolor="#0B1422",
            plot_bgcolor="#0B1422",
            font_color="#EAF2FF",
        )

        st.plotly_chart(
            fig_error,
            use_container_width=True,
        )

    else:
        st.warning("No reconstruction error values available.")

    st.divider()

    # --------------------------------------------------
    # OBSERVATION FILTERS
    # --------------------------------------------------

    st.subheader("Observation Explorer")

    col1, col2 = st.columns(2)

    with col1:
        selected_subject = st.selectbox(
            "Select Astronaut",
            ["All"] + sorted(
                df["SUBJECT_ID"].dropna().astype(str).unique()
            ),
            key="anomaly_subject",
        )

    with col2:
        selected_priority = st.selectbox(
            "Monitoring Priority",
            ["All"] + priority_order,
            key="anomaly_priority",
        )

    filtered = df.copy()

    if selected_subject != "All":
        filtered = filtered[
            filtered["SUBJECT_ID"].astype(str)
            == selected_subject
        ]

    if selected_priority != "All":
        filtered = filtered[
            filtered["MONITORING_PRIORITY"]
            == selected_priority
        ]

    display_columns = [
        "Sample Name",
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
        "ISOLATION_FOREST_STATUS",
        "AUTOENCODER_STATUS",
        "RECONSTRUCTION_ERROR",
        "MONITORING_PRIORITY_SCORE",
        "MONITORING_PRIORITY",
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(filtered)} of {total} observations."
    )

    # --------------------------------------------------
    # OBSERVATION INVESTIGATION
    # --------------------------------------------------

    st.subheader("Anomaly Investigation")

    if not filtered.empty:

        selected_index = st.selectbox(
            "Select Observation",
            filtered.index.tolist(),
            format_func=lambda i: (
                f"{filtered.loc[i, 'SUBJECT_ID']} | "
                f"{filtered.loc[i, 'TIMEPOINT']} | "
                f"{filtered.loc[i, 'MONITORING_PRIORITY']}"
            ),
            key="anomaly_observation",
        )

        selected = filtered.loc[selected_index]

        a, b, c = st.columns(3)

        a.metric(
            "Monitoring Score",
            f"{selected['MONITORING_PRIORITY_SCORE']:.0f}",
        )

        b.metric(
            "Reconstruction Error",
            f"{selected['RECONSTRUCTION_ERROR']:.4f}",
        )

        c.metric(
            "Priority",
            selected["MONITORING_PRIORITY"],
        )

        st.write(
            "**Isolation Forest:**",
            selected["ISOLATION_FOREST_STATUS"],
        )

        st.write(
            "**Autoencoder:**",
            selected["AUTOENCODER_STATUS"],
        )

        st.write("**Model Explanation:**")

        st.write(selected["EXPLANATIONS"])

    else:
        st.warning("No observations match the selected filters.")

    # --------------------------------------------------
    # EXPORT
    # --------------------------------------------------

    st.divider()

    st.subheader("Export Detection Results")

    st.download_button(
        label="Download Anomaly Detection Report",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="astrahealth_live_anomaly_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "ASTRAHEALTH AI | Experimental ML anomaly detection | "
        "Historical research observations"
    )
