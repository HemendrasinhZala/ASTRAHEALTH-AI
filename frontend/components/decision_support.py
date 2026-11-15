
import streamlit as st
import pandas as pd
import plotly.express as px

from src.models.inference_service import get_live_monitoring_data


PRIORITY_COLORS = {
    "Routine Monitoring": "#22C55E",
    "Review Recommended": "#F59E0B",
    "Priority Review": "#EF4444",
}


def render_decision_support(monitoring_df):

    st.title("Mission Decision Support")

    st.caption(
        "ML-powered monitoring insights and experimental "
        "decision support for astronaut health observations."
    )

    st.info(
        "Research prototype. Recommendations are generated "
        "from experimental model outputs and are not medical advice."
    )

    # --------------------------------------------------
    # LIVE ML INFERENCE
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
            f"Live ML analysis completed for {len(df)} observations."
        )

    except Exception as error:
        st.error(f"Live inference failed: {error}")
        return

    df["MISSION_DAY"] = pd.to_numeric(
        df["MISSION_DAY"], errors="coerce"
    )

    df["MONITORING_PRIORITY_SCORE"] = pd.to_numeric(
        df["MONITORING_PRIORITY_SCORE"], errors="coerce"
    )

    df["RECONSTRUCTION_ERROR"] = pd.to_numeric(
        df["RECONSTRUCTION_ERROR"], errors="coerce"
    )

    # --------------------------------------------------
    # MISSION DECISION SUMMARY
    # --------------------------------------------------

    st.subheader("Mission Decision Overview")

    total = len(df)

    routine = (
        df["MONITORING_PRIORITY"]
        == "Routine Monitoring"
    ).sum()

    review = (
        df["MONITORING_PRIORITY"]
        == "Review Recommended"
    ).sum()

    priority = (
        df["MONITORING_PRIORITY"]
        == "Priority Review"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Observations", total)
    c2.metric("Routine", int(routine))
    c3.metric("Review Recommended", int(review))
    c4.metric("Priority Review", int(priority))

    st.divider()

    # --------------------------------------------------
    # ASTRONAUT SELECTION
    # --------------------------------------------------

    st.subheader("Participant Decision Workspace")

    subjects = sorted(
        df["SUBJECT_ID"].dropna().astype(str).unique()
    )

    selected_subject = st.selectbox(
        "Select Astronaut",
        subjects,
        key="decision_subject",
    )

    subject_df = df[
        df["SUBJECT_ID"].astype(str)
        == selected_subject
    ].copy()

    subject_df = subject_df.sort_values("MISSION_DAY")

    if subject_df.empty:
        st.warning("No observations available.")
        return

    latest = subject_df.iloc[-1]

    # --------------------------------------------------
    # LATEST STATUS
    # --------------------------------------------------

    st.markdown(
        f"### Current Monitoring Status: {selected_subject}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Priority",
        latest["MONITORING_PRIORITY"],
    )

    col2.metric(
        "Monitoring Score",
        f"{latest['MONITORING_PRIORITY_SCORE']:.0f}/80",
    )

    col3.metric(
        "Latest Mission Day",
        int(latest["MISSION_DAY"]),
    )

    st.divider()

    # --------------------------------------------------
    # DECISION SUPPORT GUIDANCE
    # --------------------------------------------------

    st.subheader("AI-Generated Operational Guidance")

    guidance = {
        "Routine Monitoring": {
            "title": "Continue Routine Monitoring",
            "action": (
                "Continue research observation and compare "
                "future measurements with historical records."
            ),
        },

        "Review Recommended": {
            "title": "Additional Review Recommended",
            "action": (
                "Review this observation and compare it with "
                "previous participant measurements. Check the "
                "individual model indicators."
            ),
        },

        "Priority Review": {
            "title": "Priority Human Review",
            "action": (
                "Prioritize human review of this observation. "
                "Verify source data and model outputs before "
                "considering further action."
            ),
        },
    }

    latest_priority = latest["MONITORING_PRIORITY"]

    recommendation = guidance.get(
        latest_priority,
        guidance["Routine Monitoring"],
    )

    st.markdown("### " + recommendation["title"])

    if latest_priority == "Priority Review":
        st.error(recommendation["action"])

    elif latest_priority == "Review Recommended":
        st.warning(recommendation["action"])

    else:
        st.success(recommendation["action"])

    st.caption(
        "Guidance is based on experimental monitoring categories, "
        "not validated clinical thresholds."
    )

    st.divider()

    # --------------------------------------------------
    # MODEL EVIDENCE
    # --------------------------------------------------

    st.subheader("Supporting Model Evidence")

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("**Isolation Forest**")

        st.write(
            latest["ISOLATION_FOREST_STATUS"]
        )

        st.metric(
            "Isolation Forest Score",
            f"{latest['ISOLATION_FOREST_SCORE']:.4f}",
        )

    with e2:
        st.markdown("**PyTorch Autoencoder**")

        st.write(
            latest["AUTOENCODER_STATUS"]
        )

        st.metric(
            "Reconstruction Error",
            f"{latest['RECONSTRUCTION_ERROR']:.5f}",
        )

    st.markdown("**Model Explanation**")

    st.write(latest["EXPLANATIONS"])

    st.divider()

    # --------------------------------------------------
    # HISTORICAL DECISION TREND
    # --------------------------------------------------

    st.subheader("Historical Monitoring Trend")

    fig = px.line(
        subject_df,
        x="MISSION_DAY",
        y="MONITORING_PRIORITY_SCORE",
        markers=True,
        title=f"Experimental Monitoring Score — {selected_subject}",
    )

    fig.update_traces(
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=9),
    )

    fig.update_layout(
        paper_bgcolor="#0B1422",
        plot_bgcolor="#0B1422",
        font_color="#EAF2FF",
        xaxis_title="Mission Day",
        yaxis_title="Monitoring Priority Score",
        yaxis_range=[0, 80],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # DECISION HISTORY
    # --------------------------------------------------

    st.subheader("Participant Decision History")

    display_columns = [
        "TIMEPOINT",
        "MISSION_DAY",
        "ISOLATION_FOREST_STATUS",
        "AUTOENCODER_STATUS",
        "RECONSTRUCTION_ERROR",
        "MONITORING_PRIORITY_SCORE",
        "MONITORING_PRIORITY",
    ]

    st.dataframe(
        subject_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Decision Support Report",
        data=subject_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_subject}_live_decision_support.csv",
        mime="text/csv",
        use_container_width=True,
    )
