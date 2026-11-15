
import streamlit as st
import pandas as pd
import plotly.express as px


PRIORITY_COLORS = {
    "Routine Monitoring": "#22C55E",
    "Review Recommended": "#F59E0B",
    "Priority Review": "#EF4444",
}


def render_mission_timeline(monitoring_df):

    st.title("Mission Timeline")
    st.caption(
        "Historical astronaut health observations across "
        "the Inspiration4 mission timeline."
    )

    st.info(
        "Historical research data from NASA OSDR. "
        "Mission days represent recorded observation dates, "
        "not live mission telemetry."
    )

    required = [
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
        "MONITORING_PRIORITY",
        "MONITORING_PRIORITY_SCORE",
    ]

    missing = [c for c in required if c not in monitoring_df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        return

    df = monitoring_df.copy()

    df["MISSION_DAY"] = pd.to_numeric(
        df["MISSION_DAY"], errors="coerce"
    )

    df["MONITORING_PRIORITY_SCORE"] = pd.to_numeric(
        df["MONITORING_PRIORITY_SCORE"], errors="coerce"
    )

    df = df.dropna(subset=["MISSION_DAY"])

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    st.subheader("Mission Timeline Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Astronauts", df["SUBJECT_ID"].nunique())
    c2.metric("Observations", len(df))
    c3.metric("First Mission Day", int(df["MISSION_DAY"].min()))
    c4.metric("Last Mission Day", int(df["MISSION_DAY"].max()))

    st.divider()

    # --------------------------------------------------
    # FILTERS
    # --------------------------------------------------

    st.subheader("Timeline Filters")

    col1, col2 = st.columns(2)

    with col1:
        subjects = sorted(df["SUBJECT_ID"].dropna().unique())

        selected_subjects = st.multiselect(
            "Select Astronauts",
            subjects,
            default=subjects,
            key="timeline_subjects",
        )

    with col2:
        priorities = sorted(
            df["MONITORING_PRIORITY"].dropna().unique()
        )

        selected_priorities = st.multiselect(
            "Select Monitoring Priorities",
            priorities,
            default=priorities,
            key="timeline_priorities",
        )

    filtered = df[
        df["SUBJECT_ID"].isin(selected_subjects)
        & df["MONITORING_PRIORITY"].isin(selected_priorities)
    ].copy()

    if filtered.empty:
        st.warning("No observations match the selected filters.")
        return

    # --------------------------------------------------
    # MISSION EVENT TIMELINE
    # --------------------------------------------------

    st.subheader("Historical Mission Event Timeline")

    fig = px.scatter(
        filtered,
        x="MISSION_DAY",
        y="SUBJECT_ID",
        color="MONITORING_PRIORITY",
        symbol="MONITORING_PRIORITY",
        hover_data=[
            "TIMEPOINT",
            "MONITORING_PRIORITY_SCORE",
        ],
        color_discrete_map=PRIORITY_COLORS,
        title="Astronaut Health Observations by Mission Day",
    )

    fig.update_traces(marker=dict(size=14))

    fig.update_layout(
        paper_bgcolor="#0B1422",
        plot_bgcolor="#0B1422",
        font_color="#EAF2FF",
        xaxis_title="Mission Day",
        yaxis_title="Astronaut",
        height=430,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # PRIORITY TREND
    # --------------------------------------------------

    st.subheader("Monitoring Score Timeline")

    score_fig = px.line(
        filtered.sort_values("MISSION_DAY"),
        x="MISSION_DAY",
        y="MONITORING_PRIORITY_SCORE",
        color="SUBJECT_ID",
        markers=True,
        hover_data=["TIMEPOINT", "MONITORING_PRIORITY"],
        title="Historical Monitoring Priority Scores",
    )

    score_fig.update_layout(
        paper_bgcolor="#0B1422",
        plot_bgcolor="#0B1422",
        font_color="#EAF2FF",
        xaxis_title="Mission Day",
        yaxis_title="Monitoring Priority Score",
        height=400,
    )

    st.plotly_chart(
        score_fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # OBSERVATION HISTORY
    # --------------------------------------------------

    st.subheader("Mission Observation History")

    display_columns = [
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
        "MONITORING_PRIORITY_SCORE",
        "MONITORING_PRIORITY",
    ]

    history = filtered.sort_values("MISSION_DAY")

    st.dataframe(
        history[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # EXPORT
    # --------------------------------------------------

    st.download_button(
        "Download Mission Timeline",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="astrahealth_mission_timeline.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "ASTRAHEALTH AI | Historical mission timeline "
        "| Experimental research prototype"
    )
