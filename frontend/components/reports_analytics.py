
import streamlit as st
import pandas as pd
import plotly.express as px

from src.models.inference_service import get_live_monitoring_data


PRIORITY_COLORS = {
    "Routine Monitoring": "#22C55E",
    "Review Recommended": "#F59E0B",
    "Priority Review": "#EF4444",
}


def render_reports_analytics(monitoring_df, health_df):

    st.title("Reports & Analytics")

    st.caption(
        "Mission-wide astronaut analytics using actual "
        "experimental ML inference and historical health data."
    )

    st.info(
        "All measurements are historical NASA OSDR research data. "
        "Model outputs are experimental and are not clinical diagnoses."
    )

    # --------------------------------------------------
    # LIVE MODEL INFERENCE
    # --------------------------------------------------

    st.subheader("Live Analytics")

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
            f"Live ML analytics generated for {len(df)} observations."
        )

    except Exception as error:
        st.error(f"Live analytics failed: {error}")
        return

    health = health_df.copy()

    df["MISSION_DAY"] = pd.to_numeric(
        df["MISSION_DAY"], errors="coerce"
    )

    df["MONITORING_PRIORITY_SCORE"] = pd.to_numeric(
        df["MONITORING_PRIORITY_SCORE"], errors="coerce"
    )

    # --------------------------------------------------
    # KPI SUMMARY
    # --------------------------------------------------

    st.subheader("Mission Analytics Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Astronauts",
        df["SUBJECT_ID"].nunique(),
    )

    c2.metric(
        "Observations",
        len(df),
    )

    c3.metric(
        "Review Observations",
        int(
            df["MONITORING_PRIORITY"].isin([
                "Review Recommended",
                "Priority Review",
            ]).sum()
        ),
    )

    c4.metric(
        "Priority Reviews",
        int(
            (
                df["MONITORING_PRIORITY"]
                == "Priority Review"
            ).sum()
        ),
    )

    st.divider()

    # --------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Live Monitoring Priority Distribution")

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

    fig = px.pie(
        priority_counts,
        names="Priority",
        values="Observations",
        hole=0.55,
        title="Current Model-Generated Priority Breakdown",
        color="Priority",
        color_discrete_map=PRIORITY_COLORS,
    )

    fig.update_layout(
        paper_bgcolor="#0B1422",
        font_color="#EAF2FF",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # ASTRONAUT COMPARISON
    # --------------------------------------------------

    st.subheader("Astronaut Comparison")

    comparison = (
        df.groupby("SUBJECT_ID")
        .agg(
            Observations=("TIMEPOINT", "count"),
            Average_Score=(
                "MONITORING_PRIORITY_SCORE",
                "mean",
            ),
            Priority_Reviews=(
                "MONITORING_PRIORITY",
                lambda x: (x == "Priority Review").sum(),
            ),
            Review_Recommended=(
                "MONITORING_PRIORITY",
                lambda x: (x == "Review Recommended").sum(),
            ),
        )
        .reset_index()
    )

    comparison["Average_Score"] = (
        comparison["Average_Score"].round(2)
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        comparison,
        x="SUBJECT_ID",
        y="Average_Score",
        color="SUBJECT_ID",
        title="Average Experimental Monitoring Score by Participant",
        labels={
            "SUBJECT_ID": "Astronaut",
            "Average_Score": "Average Monitoring Score",
        },
    )

    fig.update_layout(
        paper_bgcolor="#0B1422",
        plot_bgcolor="#0B1422",
        font_color="#EAF2FF",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # MODEL STATUS ANALYSIS
    # --------------------------------------------------

    st.subheader("AI Model Monitoring Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Isolation Forest Status**")

        if_counts = (
            df["ISOLATION_FOREST_STATUS"]
            .fillna("Unavailable")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )

        st.dataframe(
            if_counts,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("**Autoencoder Status**")

        ae_counts = (
            df["AUTOENCODER_STATUS"]
            .fillna("Unavailable")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )

        st.dataframe(
            ae_counts,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------
    # ANOMALY SUMMARY
    # --------------------------------------------------

    if_flags = (
        df["ISOLATION_FOREST_STATUS"]
        .astype(str).str.strip().eq("Anomaly Flag").sum()
    )

    ae_flags = (
        df["AUTOENCODER_STATUS"]
        .astype(str).str.strip().eq("Anomaly Flag").sum()
    )

    st.markdown("**Exact Anomaly Flag Counts**")

    a, b = st.columns(2)

    a.metric("Isolation Forest Flags", int(if_flags))
    b.metric("Autoencoder Flags", int(ae_flags))

    st.divider()

    # --------------------------------------------------
    # HISTORICAL SCORE TREND
    # --------------------------------------------------

    st.subheader("Monitoring Score Analysis")

    trend = df.sort_values("MISSION_DAY")

    fig = px.line(
        trend,
        x="MISSION_DAY",
        y="MONITORING_PRIORITY_SCORE",
        color="SUBJECT_ID",
        markers=True,
        title="Experimental Monitoring Score Across Mission Days",
        hover_data=[
            "TIMEPOINT",
            "MONITORING_PRIORITY",
        ],
    )

    fig.update_layout(
        paper_bgcolor="#0B1422",
        plot_bgcolor="#0B1422",
        font_color="#EAF2FF",
        yaxis_range=[0, 80],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # HEALTH DATA SUMMARY
    # --------------------------------------------------

    st.subheader("Historical Health Measurement Summary")

    preferred = [
        "hemoglobin_value_percent",
        "hematocrit_value_percent",
        "platelet_count_value_thousand_per_microliter",
        "white_blood_cell_count_value_thousand_per_microliter",
        "red_blood_cell_count_value_million_per_microliter",
    ]

    available = [
        column
        for column in preferred
        if column in health.columns
    ]

    if available:

        selected_measurement = st.selectbox(
            "Select Measurement",
            available,
            key="analytics_measurement",
        )

        values = pd.to_numeric(
            health[selected_measurement],
            errors="coerce",
        )

        summary = pd.DataFrame({
            "Statistic": [
                "Valid observations",
                "Mean",
                "Minimum",
                "Maximum",
                "Standard deviation",
            ],
            "Value": [
                int(values.count()),
                round(values.mean(), 3),
                round(values.min(), 3),
                round(values.max(), 3),
                round(values.std(), 3),
            ],
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning(
            "No matching numeric health measurements found."
        )

    st.divider()

    # --------------------------------------------------
    # REPORT EXPORT
    # --------------------------------------------------

    st.subheader("Export Mission Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Live Monitoring Report",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="astrahealth_live_monitoring_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Download Astronaut Comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="astrahealth_astronaut_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "ASTRAHEALTH AI | Live Experimental Analytics | Research Prototype"
    )
