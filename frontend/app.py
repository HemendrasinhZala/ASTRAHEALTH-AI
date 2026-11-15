
from pathlib import Path
import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.system_information import (
    render_system_information,
)

from frontend.components.reports_analytics import (
    render_reports_analytics,
)

from frontend.components.health_assistant import (
    render_health_assistant,
)

from frontend.components.mission_timeline import (
    render_mission_timeline,
)

from frontend.components.agentic_ai import (
    render_agentic_ai,
)

from frontend.components.decision_support import (
    render_decision_support,
)

from frontend.components.decision_support import (
    render_decision_support,
)

from frontend.components.data_loader import (
    load_all_data,
    get_dashboard_summary,
)

from frontend.components.astronaut_monitoring import (
    render_astronaut_monitoring,
)

from frontend.components.anomaly_detection import (
    render_anomaly_detection,
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="ASTRAHEALTH AI | Mission Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSS_PATH = PROJECT_ROOT / "frontend" / "styles" / "main.css"

AUTOENCODER_PATH = (
    PROJECT_ROOT / "artifacts" / "models"
    / "astronaut_autoencoder.pth"
)

ISOLATION_FOREST_PATH = (
    PROJECT_ROOT / "artifacts" / "models"
    / "isolation_forest.joblib"
)


# ==========================================================
# 2. HTML AND CSS HELPERS
# ==========================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


def load_css():
    if CSS_PATH.exists():
        css_content = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Custom stylesheet not found.")


load_css()


# ==========================================================
# 3. LOAD DATA
# ==========================================================

try:
    monitoring_df, health_df = load_all_data()
    summary = get_dashboard_summary(monitoring_df)

except Exception as error:
    st.error(f"Dashboard data loading failed: {error}")
    st.stop()


# ==========================================================
# 4. SESSION STATE
# ==========================================================

if "selected_astronaut" not in st.session_state:
    st.session_state.selected_astronaut = "C001"


def select_astronaut(subject_id):
    st.session_state.selected_astronaut = subject_id


# ==========================================================
# 5. SIDEBAR NAVIGATION
# ==========================================================

pages = {
    "Agentic AI Control Center": "◉",
    "Mission Overview": "◈",
    "Astronaut Monitoring": "♙",
    "AI Anomaly Detection": "⌁",
    "Decision Support": "◇",
    "Mission Timeline": "◷",
    "AI Health Assistant": "✧",
    "Reports & Analytics": "▤",
    "System Information": "⚙",
}

with st.sidebar:

    render_html("""
        <div class="brand-container">
            <div class="brand-icon">✦</div>
            <div class="brand-name">
                ASTRAHEALTH <span>AI</span>
            </div>
            <div class="brand-subtitle">
                MISSION CONTROL SYSTEM
            </div>
        </div>
    """)

    render_html("""
        <div class="sidebar-label">
            MISSION OPERATIONS
        </div>
    """)

    selected_page = st.radio(
        "Navigation",
        options=list(pages.keys()),
        format_func=lambda page: (
            f"{pages[page]}   {page}"
        ),
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.markdown("---")

    render_html("""
        <div class="sidebar-footer">
            <div class="status-dot"></div>
            APPLICATION ONLINE

            <div class="footer-version">
                ASTRAHEALTH AI · v1.0
            </div>
        </div>
    """)


# ==========================================================
# 6. MAIN HEADER
# ==========================================================

render_html("""
    <div class="top-header">
        <div>
            <div class="eyebrow">
                NASA SPACE HEALTH MONITORING CONCEPT
            </div>

            <div class="main-heading">
                MISSION CONTROL
            </div>

            <div class="sub-heading">
                Astronaut Health Monitoring & Decision Support
            </div>
        </div>

        <div class="header-status">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>
    </div>
""")

render_html("""
    <div class="data-notice">
        <strong>DATA MODE:</strong>
        Historical NASA OSDR research data ·
        Not live astronaut telemetry
    </div>
""")


# ==========================================================
# 7. PAGE ROUTING
# ==========================================================

if selected_page == "Mission Overview":

    # ------------------------------------------------------
    # MISSION OVERVIEW HEADER
    # ------------------------------------------------------

    render_html("""
        <div class="section-heading">
            MISSION OVERVIEW
        </div>

        <div class="section-description">
            Central monitoring workspace and mission health summary.
        </div>
    """)

    # ------------------------------------------------------
    # MISSION STATISTICS
    # ------------------------------------------------------

    st.markdown("### Mission Statistics")

    kpis = [
        (
            "ASTRONAUTS",
            summary["astronaut_count"],
            "Participants",
        ),
        (
            "OBSERVATIONS",
            summary["observation_count"],
            "Health records",
        ),
        (
            "REVIEW FLAGS",
            summary["review_count"],
            "Experimental flags",
        ),
        (
            "MODELS AVAILABLE",
            f'{summary["model_count"]}/2',
            "Saved model artifacts",
        ),
    ]

    columns = st.columns(4)

    for column, (label, value, description) in zip(columns, kpis):
        with column:
            render_html(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-description">{description}</div>
                </div>
            """)

    # ------------------------------------------------------
    # PRIORITY SUMMARY
    # ------------------------------------------------------

    st.markdown("### Monitoring Priority Summary")

    priority_order = [
        "Routine Monitoring",
        "Review Recommended",
        "Priority Review",
    ]

    priority_colors = {
        "Routine Monitoring": "#22C55E",
        "Review Recommended": "#F59E0B",
        "Priority Review": "#EF4444",
    }

    priority_counts = (
        monitoring_df["MONITORING_PRIORITY"]
        .value_counts()
        .reindex(priority_order, fill_value=0)
    )

    priority_columns = st.columns(3)

    priority_styles = {
        "Routine Monitoring": "normal",
        "Review Recommended": "warning",
        "Priority Review": "critical",
    }

    for column, priority in zip(priority_columns, priority_order):

        count = int(priority_counts[priority])

        with column:
            render_html(f"""
                <div class="priority-card {priority_styles[priority]}">
                    <div class="metric-label">{priority.upper()}</div>
                    <div class="priority-value">{count}</div>
                    <div class="metric-description">
                        Historical model-generated observations
                    </div>
                </div>
            """)

    st.caption(
        "Experimental model-generated monitoring categories. "
        "These are not clinical diagnoses."
    )

    # ------------------------------------------------------
    # PRIORITY DONUT CHART
    # ------------------------------------------------------

    st.markdown("### Monitoring Distribution Analysis")

    donut_df = pd.DataFrame({
        "Priority": priority_counts.index,
        "Observations": priority_counts.values,
    })

    left, right = st.columns([1.4, 1])

    with left:

        fig_donut = px.pie(
            donut_df,
            names="Priority",
            values="Observations",
            hole=0.68,
            color="Priority",
            color_discrete_map=priority_colors,
        )

        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
        )

        fig_donut.update_layout(
            paper_bgcolor="#101E31",
            plot_bgcolor="#101E31",
            font_color="#E6F0FC",
            showlegend=True,
            height=370,
            annotations=[
                dict(
                    text=f"<b>{len(monitoring_df)}</b><br>OBSERVATIONS",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="#E6F0FC"),
                )
            ],
        )

        st.plotly_chart(
            fig_donut,
            use_container_width=True,
            key="mission_priority_donut",
        )

    with right:

        st.markdown("#### Priority Breakdown")

        total_observations = len(monitoring_df)

        for priority in priority_order:

            count = int(priority_counts[priority])

            percentage = (
                count / total_observations * 100
                if total_observations > 0
                else 0
            )

            color = priority_colors[priority]

            render_html(f"""
                <div style="
                    background:#101E31;
                    border:1px solid #223752;
                    border-left:4px solid {color};
                    border-radius:8px;
                    padding:17px;
                    margin-bottom:12px;
                ">
                    <div style="color:#8BA2BD;font-size:10px;font-weight:700;">
                        {priority.upper()}
                    </div>
                    <div style="color:#E6F0FC;font-size:27px;font-weight:800;margin-top:8px;">
                        {count}
                    </div>
                    <div style="color:#8BA2BD;font-size:11px;">
                        {percentage:.2f}% of observations
                    </div>
                </div>
            """)

    # ------------------------------------------------------
    # ASTRONAUT PARTICIPANT CARDS
    # ------------------------------------------------------

    st.markdown("### Astronaut Monitoring")

    astronaut_ids = sorted(
        monitoring_df["SUBJECT_ID"].dropna().unique()
    )

    astronaut_columns = st.columns(4)

    for column, subject_id in zip(astronaut_columns, astronaut_ids):

        subject_data = monitoring_df[
            monitoring_df["SUBJECT_ID"] == subject_id
        ]

        routine_count = (
            subject_data["MONITORING_PRIORITY"]
            == "Routine Monitoring"
        ).sum()

        review_count = (
            subject_data["MONITORING_PRIORITY"].isin([
                "Review Recommended",
                "Priority Review",
            ])
        ).sum()

        with column:

            render_html(f"""
                <div class="astronaut-card">
                    <div class="astronaut-label">
                        ASTRONAUT PARTICIPANT
                    </div>
                    <div class="astronaut-id">{subject_id}</div>
                    <div class="astronaut-stat">
                        {len(subject_data)} observations
                    </div>
                    <div class="astronaut-stat">
                        {routine_count} routine · {review_count} review flags
                    </div>
                </div>
            """)

            st.button(
                "VIEW PROFILE →",
                key=f"profile_{subject_id}",
                use_container_width=True,
                on_click=select_astronaut,
                args=(subject_id,),
            )

    # ------------------------------------------------------
    # SELECTED ASTRONAUT PROFILE
    # ------------------------------------------------------

    selected_id = st.session_state.selected_astronaut

    selected_monitoring = monitoring_df[
        monitoring_df["SUBJECT_ID"] == selected_id
    ].copy()

    selected_health = health_df[
        health_df["SUBJECT_ID"] == selected_id
    ].copy()

    st.markdown(f"### Selected Participant: {selected_id}")

    if not selected_monitoring.empty:

        latest_record = selected_monitoring.sort_values(
            "MISSION_DAY"
        ).iloc[-1]

        profile_columns = st.columns(3)

        profile_columns[0].metric(
            "Total Observations",
            len(selected_monitoring),
        )

        profile_columns[1].metric(
            "Latest Mission Day",
            int(latest_record["MISSION_DAY"]),
        )

        profile_columns[2].metric(
            "Latest Monitoring Priority",
            latest_record["MONITORING_PRIORITY"],
        )

        st.markdown("#### Historical Health Measurements")

        health_display_columns = [
            "TIMEPOINT",
            "MISSION_DAY",
            "hemoglobin_value_percent",
            "hematocrit_value_percent",
            "white_blood_cell_count_value_thousand_per_microliter",
            "platelet_count_value_thousand_per_microliter",
        ]

        display_df = selected_health[
            health_display_columns
        ].copy()

        display_df = display_df.rename(columns={
            "hemoglobin_value_percent": "Hemoglobin",
            "hematocrit_value_percent": "Hematocrit",
            "white_blood_cell_count_value_thousand_per_microliter": "WBC",
            "platelet_count_value_thousand_per_microliter": "Platelets",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------
    # INTERACTIVE HEALTH TELEMETRY
    # ------------------------------------------------------

    st.markdown("### Interactive Health Telemetry")

    st.caption(
        "Explore historical CBC measurements by astronaut "
        "and measurement type."
    )

    telemetry_columns = {
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
    }

    telemetry_left, telemetry_right = st.columns(2)

    available_astronauts = sorted(
        health_df["SUBJECT_ID"].dropna().unique()
    )

    with telemetry_left:

        selected_index = (
            available_astronauts.index(selected_id)
            if selected_id in available_astronauts
            else 0
        )

        telemetry_astronaut = st.selectbox(
            "Select Astronaut",
            options=available_astronauts,
            index=selected_index,
            key="telemetry_astronaut",
        )

    with telemetry_right:

        selected_measurement = st.selectbox(
            "Select Health Measurement",
            options=list(telemetry_columns.keys()),
            key="telemetry_measurement",
        )

    measurement_column = telemetry_columns[selected_measurement]

    chart_data = health_df[
        health_df["SUBJECT_ID"] == telemetry_astronaut
    ].copy()

    chart_data = chart_data.sort_values("MISSION_DAY")

    chart_data[measurement_column] = pd.to_numeric(
        chart_data[measurement_column],
        errors="coerce",
    )

    chart_data = chart_data.dropna(
        subset=["MISSION_DAY", measurement_column]
    )

    if chart_data.empty:

        st.warning("No valid measurements available for this selection.")

    else:

        fig_telemetry = px.line(
            chart_data,
            x="MISSION_DAY",
            y=measurement_column,
            markers=True,
            hover_data=["TIMEPOINT", "Sample Name"],
            title=f"{selected_measurement} — {telemetry_astronaut}",
        )

        fig_telemetry.update_traces(
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=9, color="#22D3EE"),
        )

        fig_telemetry.update_layout(
            paper_bgcolor="#101E31",
            plot_bgcolor="#101E31",
            font_color="#E6F0FC",
            xaxis_title="Mission Day",
            yaxis_title=selected_measurement,
            height=420,
        )

        st.plotly_chart(
            fig_telemetry,
            use_container_width=True,
            key="health_telemetry_chart",
        )

        st.caption(
            f"Showing {len(chart_data)} available observations "
            f"for {telemetry_astronaut}."
        )

    # ------------------------------------------------------
    # RECENT MONITORING EVENTS
    # ------------------------------------------------------

    st.markdown("### Recent Monitoring Events")

    recent_events = monitoring_df.sort_values(
        "MISSION_DAY",
        ascending=False,
    ).head(8)

    event_columns = [
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
        "MONITORING_PRIORITY",
        "MONITORING_PRIORITY_SCORE",
    ]

    st.dataframe(
        recent_events[event_columns],
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------
    # AI MODEL STATUS
    # ------------------------------------------------------

    st.markdown("### AI Model Status")

    model_columns = st.columns(2)

    model_information = [
        (
            "PYTORCH AUTOENCODER",
            AUTOENCODER_PATH.exists(),
            "Unsupervised neural network",
            "Reconstruction-based anomaly detection",
        ),
        (
            "ISOLATION FOREST",
            ISOLATION_FOREST_PATH.exists(),
            "Scikit-learn ensemble model",
            "Isolation-based anomaly detection",
        ),
    ]

    for column, information in zip(model_columns, model_information):

        model_name, available, model_type, description = information

        with column:

            if available:
                status_text = "ARTIFACT AVAILABLE"
                status_color = "#22C55E"
            else:
                status_text = "ARTIFACT MISSING"
                status_color = "#EF4444"

            render_html(f"""
                <div class="panel">
                    <div class="panel-title">{model_name}</div>
                    <div style="
                        color:{status_color};
                        font-size:11px;
                        font-weight:700;
                        margin-top:15px;
                    ">
                        ● {status_text}
                    </div>
                    <div class="metric-description">{model_type}</div>
                    <div class="metric-description">{description}</div>
                </div>
            """)

    st.caption(
        "Artifact availability confirms that saved model files exist. "
        "It does not independently verify runtime inference or "
        "clinical validity."
    )


# ==========================================================
# 8. ASTRONAUT MONITORING
# ==========================================================

elif selected_page == "Astronaut Monitoring":

    render_astronaut_monitoring(
        monitoring_df,
        health_df,
    )


# ==========================================================
# 9. AI ANOMALY DETECTION
# ==========================================================

elif selected_page == "AI Anomaly Detection":

    render_anomaly_detection(monitoring_df)
elif selected_page == "Decision Support":

    render_decision_support(monitoring_df)

elif selected_page == "Agentic AI Control Center":

    render_agentic_ai(monitoring_df)
elif selected_page == "Mission Timeline":

    render_mission_timeline(monitoring_df)

elif selected_page == "AI Health Assistant":

    render_health_assistant(
        monitoring_df,
        health_df,
    )

elif selected_page == "Reports & Analytics":

    render_reports_analytics(
        monitoring_df,
        health_df,
    )

elif selected_page == "System Information":

    render_system_information(
        monitoring_df,
        health_df,
    )

# ==========================================================
# 10. OTHER MODULES
# ==========================================================

elif selected_page in [
    "Decision Support",
    "Mission Timeline",
    "AI Health Assistant",
    "Reports & Analytics",
    "System Information",
]:

    render_html(f"""
        <div class="section-heading">
            {selected_page.upper()}
        </div>

        <div class="section-description">
            ASTRAHEALTH AI module workspace.
        </div>
    """)

    st.info(
        f"{selected_page} will be implemented "
        "in its dedicated development phase."
    )


# ==========================================================
# 11. APPLICATION FOOTER
# ==========================================================

render_html("""
    <div class="app-footer">
        ASTRAHEALTH AI · EXPERIMENTAL RESEARCH PROTOTYPE
        <br>
        For research and demonstration purposes only.
        Not a clinical system.
    </div>
""")
