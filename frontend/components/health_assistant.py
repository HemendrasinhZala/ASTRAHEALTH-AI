
import streamlit as st
import pandas as pd

from src.models.inference_service import get_live_monitoring_data


# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare_live_data(monitoring_df):

    metadata_columns = [
        "Sample Name",
        "SUBJECT_ID",
        "TIMEPOINT",
        "MISSION_DAY",
    ]

    missing = [
        column for column in metadata_columns
        if column not in monitoring_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metadata columns: {missing}"
        )

    metadata = monitoring_df[
        metadata_columns
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

    return live_df


# ==========================================================
# RESPONSE GENERATION
# ==========================================================

def generate_response(question, monitoring_df, health_df):

    q = question.lower().strip()

    df = monitoring_df.copy()
    health = health_df.copy()

    if not q:
        return "Please enter a question about the available astronaut observations."

    # ------------------------------------------------------
    # GENERAL QUESTIONS
    # ------------------------------------------------------

    if any(word in q.split() for word in ["hello", "hi", "hey"]):
        return (
            "Hello! I'm the ASTRAHEALTH AI Health Assistant. "
            "I can explore historical astronaut health records, "
            "actual ML model outputs and experimental monitoring priorities."
        )

    if "help" in q or "what can you do" in q:
        return (
            "I can help you explore ASTRAHEALTH AI data.\n\n"
            "- Count astronauts and observations.\n"
            "- Review current experimental monitoring priorities.\n"
            "- Explain Autoencoder and Isolation Forest outputs.\n"
            "- Retrieve an astronaut's latest observation.\n"
            "- Summarize historical blood measurements.\n"
            "- Compare participants' monitoring results.\n\n"
            "All answers are based on the available historical research dataset."
        )

    if (
        "how many astronaut" in q
        or "number of astronaut" in q
        or "total astronaut" in q
    ):
        return (
            f"The dataset contains "
            f"{df['SUBJECT_ID'].nunique()} astronaut participants."
        )

    if (
        "how many observation" in q
        or "total observation" in q
        or "number of observation" in q
    ):
        return (
            f"The dataset contains {len(df)} historical observations."
        )

    # ------------------------------------------------------
    # EXACT MODEL ANOMALY COUNTS
    # ------------------------------------------------------

    if (
        "anomaly count" in q
        or "how many anomaly" in q
        or "anomaly flags" in q
    ):
        if_flags = (
            df["ISOLATION_FOREST_STATUS"]
            .astype(str)
            .str.strip()
            .eq("Anomaly Flag")
            .sum()
        )

        ae_flags = (
            df["AUTOENCODER_STATUS"]
            .astype(str)
            .str.strip()
            .eq("Anomaly Flag")
            .sum()
        )

        return (
            "Actual model inference results:\n\n"
            f"- Isolation Forest anomaly flags: {int(if_flags)}\n"
            f"- Autoencoder anomaly flags: {int(ae_flags)}\n"
            f"- Total observations analyzed: {len(df)}\n\n"
            "These are experimental anomaly indicators, not clinical diagnoses."
        )

    # ------------------------------------------------------
    # PRIORITY DISTRIBUTION
    # ------------------------------------------------------

    if (
        "priority" in q
        or "monitoring distribution" in q
    ):
        counts = df["MONITORING_PRIORITY"].value_counts()

        return (
            "Live model-generated monitoring priority distribution:\n\n"
            + "\n".join(
                f"- {name}: {int(counts.get(name, 0))} observations"
                for name in [
                    "Routine Monitoring",
                    "Review Recommended",
                    "Priority Review",
                ]
            )
            + "\n\nThese are experimental monitoring categories, "
            "not clinical diagnoses."
        )

    # ------------------------------------------------------
    # ISOLATION FOREST
    # ------------------------------------------------------

    if "isolation forest" in q:

        counts = (
            df["ISOLATION_FOREST_STATUS"]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        return (
            "Isolation Forest live prediction summary:\n\n"
            + "\n".join(
                f"- {status}: {count}"
                for status, count in counts.items()
            )
            + "\n\nThe model identifies observations that differ "
            "from learned data patterns. Its flags are not diagnoses."
        )

    # ------------------------------------------------------
    # AUTOENCODER
    # ------------------------------------------------------

    if "autoencoder" in q or "reconstruction" in q:

        counts = (
            df["AUTOENCODER_STATUS"]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        mean_error = df["RECONSTRUCTION_ERROR"].mean()
        max_error = df["RECONSTRUCTION_ERROR"].max()

        return (
            "PyTorch Autoencoder live prediction summary:\n\n"
            + "\n".join(
                f"- {status}: {count}"
                for status, count in counts.items()
            )
            + f"\n\n- Mean reconstruction error: {mean_error:.4f}"
            + f"\n- Maximum reconstruction error: {max_error:.4f}"
            + "\n\nReconstruction error measures how differently "
            "an observation is reconstructed by the trained model."
        )

    # ------------------------------------------------------
    # ASTRONAUT-SPECIFIC QUESTIONS
    # ------------------------------------------------------

    subjects = (
        df["SUBJECT_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_subject = next(
        (
            subject
            for subject in subjects
            if subject.lower() in q
        ),
        None,
    )

    if selected_subject:

        subject_df = df[
            df["SUBJECT_ID"].astype(str)
            == selected_subject
        ].copy()

        subject_df = subject_df.sort_values(
            "MISSION_DAY"
        )

        latest = subject_df.iloc[-1]

        if any(
            word in q
            for word in ["latest", "current", "recent"]
        ):
            return (
                f"Latest recorded observation for {selected_subject}:\n\n"
                f"- Sample: {latest['Sample Name']}\n"
                f"- Timepoint: {latest['TIMEPOINT']}\n"
                f"- Mission day: {latest['MISSION_DAY']}\n"
                f"- Isolation Forest: {latest['ISOLATION_FOREST_STATUS']}\n"
                f"- Autoencoder: {latest['AUTOENCODER_STATUS']}\n"
                f"- Monitoring priority: {latest['MONITORING_PRIORITY']}\n"
                f"- Monitoring score: "
                f"{latest['MONITORING_PRIORITY_SCORE']:.0f}/80\n"
                f"- Reconstruction error: "
                f"{latest['RECONSTRUCTION_ERROR']:.4f}\n\n"
                "This is the latest available historical observation, "
                "not the astronaut's current health status."
            )

        if "how many" in q or "observations" in q:
            return (
                f"{selected_subject} has {len(subject_df)} "
                "historical observations."
            )

        if "explanation" in q or "why" in q:
            explanations = latest["EXPLANATIONS"]

            if isinstance(explanations, list):
                explanation = "\n".join(
                    f"- {item}" for item in explanations
                )
            else:
                explanation = str(explanations)

            return (
                f"Model explanation for {selected_subject}'s "
                "latest observation:\n\n"
                f"{explanation}\n\n"
                "These explanations describe model behavior, "
                "not medical causes."
            )

        if "priority" in q or "score" in q:
            return (
                f"Monitoring results for {selected_subject}:\n\n"
                f"- Priority: {latest['MONITORING_PRIORITY']}\n"
                f"- Experimental score: "
                f"{latest['MONITORING_PRIORITY_SCORE']:.0f}/80\n"
                f"- Isolation Forest: "
                f"{latest['ISOLATION_FOREST_STATUS']}\n"
                f"- Autoencoder: {latest['AUTOENCODER_STATUS']}"
            )

        return (
            f"Participant: {selected_subject}\n\n"
            f"- Observations: {len(subject_df)}\n"
            f"- Latest mission day: {latest['MISSION_DAY']}\n"
            f"- Latest priority: {latest['MONITORING_PRIORITY']}\n"
            f"- Latest score: "
            f"{latest['MONITORING_PRIORITY_SCORE']:.0f}/80\n"
            f"- Isolation Forest: "
            f"{latest['ISOLATION_FOREST_STATUS']}\n"
            f"- Autoencoder: {latest['AUTOENCODER_STATUS']}"
        )

    # ------------------------------------------------------
    # HISTORICAL HEALTH MEASUREMENTS
    # ------------------------------------------------------

    measurement_map = {
        "hemoglobin": "hemoglobin_value_percent",
        "hematocrit": "hematocrit_value_percent",
        "platelet": "platelet_count_value_thousand_per_microliter",
        "white blood cell":
            "white_blood_cell_count_value_thousand_per_microliter",
        "red blood cell":
            "red_blood_cell_count_value_million_per_microliter",
    }

    for measurement, column in measurement_map.items():

        if measurement in q:

            if column not in health.columns:
                return (
                    "This measurement is not available in the dataset."
                )

            values = pd.to_numeric(
                health[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                return "No valid measurements are available."

            return (
                f"Historical {measurement.title()} summary:\n\n"
                f"- Valid records: {len(values)}\n"
                f"- Average: {values.mean():.3f}\n"
                f"- Minimum: {values.min():.3f}\n"
                f"- Maximum: {values.max():.3f}\n\n"
                "These are descriptive statistics, not medical "
                "reference ranges or diagnostic assessments."
            )

    # ------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------

    return (
        "I couldn't identify that specific question in the available data.\n\n"
        "Try asking:\n"
        "- How many astronauts are monitored?\n"
        "- Show monitoring priority distribution.\n"
        "- How many anomaly flags were detected?\n"
        "- What is the latest observation for C001?\n"
        "- Explain C003's latest observation.\n"
        "- What is the Isolation Forest status?\n"
        "- Explain Autoencoder reconstruction error.\n"
        "- Show average historical hemoglobin.\n\n"
        "I can only answer using available historical research data."
    )


# ==========================================================
# STREAMLIT HEALTH ASSISTANT
# ==========================================================

def render_health_assistant(monitoring_df, health_df):

    st.title("AI Health Assistant")

    st.caption(
        "Conversational exploration of historical astronaut "
        "health observations and actual experimental ML outputs."
    )

    st.info(
        "Research assistant: Responses are generated from "
        "historical records and trained model predictions. "
        "This is not a medical diagnostic or emergency-response system."
    )

    # ------------------------------------------------------
    # LIVE MODEL DATA
    # ------------------------------------------------------

    try:
        live_df = prepare_live_data(monitoring_df)

    except Exception as error:
        st.error(
            f"Unable to prepare live model data: {error}"
        )
        return

    # ------------------------------------------------------
    # SESSION STATE
    # ------------------------------------------------------

    if "health_chat_history" not in st.session_state:
        st.session_state.health_chat_history = []

    # ------------------------------------------------------
    # ASSISTANT STATUS
    # ------------------------------------------------------

    st.subheader("Assistant Capabilities")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Astronauts",
        live_df["SUBJECT_ID"].nunique(),
    )

    c2.metric(
        "Observations",
        len(live_df),
    )

    c3.metric(
        "ML Models",
        "2 Online",
    )

    c4.metric(
        "Data Source",
        "NASA OSDR",
    )

    st.divider()

    # ------------------------------------------------------
    # SUGGESTED QUESTIONS
    # ------------------------------------------------------

    st.subheader("Suggested Questions")

    suggestions = [
        "How many astronauts are monitored?",
        "Show monitoring priority distribution.",
        "How many anomaly flags were detected?",
        "What is the latest observation for C001?",
        "Explain C003's latest observation.",
        "What is the Isolation Forest status?",
        "Explain Autoencoder reconstruction error.",
        "Show average historical hemoglobin.",
    ]

    cols = st.columns(2)

    for i, question in enumerate(suggestions):

        with cols[i % 2]:

            if st.button(
                question,
                key=f"health_suggestion_{i}",
                use_container_width=True,
            ):

                st.session_state.health_chat_history.append({
                    "role": "user",
                    "content": question,
                })

                answer = generate_response(
                    question,
                    live_df,
                    health_df,
                )

                st.session_state.health_chat_history.append({
                    "role": "assistant",
                    "content": answer,
                })

                st.rerun()

    st.divider()

    # ------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------

    st.subheader("Mission Health Conversation")

    for message in st.session_state.health_chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ------------------------------------------------------
    # CHAT INPUT
    # ------------------------------------------------------

    question = st.chat_input(
        "Ask about astronaut health observations..."
    )

    if question:

        st.session_state.health_chat_history.append({
            "role": "user",
            "content": question,
        })

        answer = generate_response(
            question,
            live_df,
            health_df,
        )

        st.session_state.health_chat_history.append({
            "role": "assistant",
            "content": answer,
        })

        st.rerun()

    # ------------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------------

    if st.button(
        "Clear Conversation",
        use_container_width=False,
    ):

        st.session_state.health_chat_history = []
        st.rerun()

    st.caption(
        "ASTRAHEALTH AI | Dataset-grounded ML-assisted "
        "conversational research prototype"
    )
