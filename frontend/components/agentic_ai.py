
import streamlit as st
import pandas as pd

from src.models.inference_service import get_live_monitoring_data
from src.agents.monitoring_agent import orchestrate_monitoring
from src.agents.agent_store import (
    initialize_store,
    save_agent_execution,
    update_approval,
    get_recent_executions,
)


def render_agentic_ai(monitoring_df):

    st.title("Agentic AI Control Center")

    st.caption(
        "ML-powered monitoring workflow with "
        "human-in-the-loop decision support."
    )

    st.info(
        "Research prototype: Uses actual Autoencoder and "
        "Isolation Forest predictions with a rule-based "
        "backend agent workflow. No spacecraft actions "
        "are executed."
    )

    # --------------------------------------------------
    # INITIALIZE PERSISTENT STORAGE
    # --------------------------------------------------

    try:
        initialize_store()
    except Exception as error:
        st.error(f"Agent database initialization failed: {error}")
        return

    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []

    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None

    if "agent_execution_id" not in st.session_state:
        st.session_state.agent_execution_id = None

    if "agent_approval" not in st.session_state:
        st.session_state.agent_approval = "Pending"

    # --------------------------------------------------
    # LIVE ML INFERENCE
    # --------------------------------------------------

    st.subheader("Live Model Inference")

    try:
        metadata_columns = [
            "Sample Name",
            "SUBJECT_ID",
            "TIMEPOINT",
            "MISSION_DAY",
        ]

        metadata_df = monitoring_df[metadata_columns].copy()
        live_df = get_live_monitoring_data(metadata_df)

        st.success(
            f"ML inference successful: "
            f"{len(live_df)} observations processed."
        )

    except Exception as error:
        st.error(f"ML inference failed: {error}")
        return

    # --------------------------------------------------
    # AGENT SYSTEM STATUS
    # --------------------------------------------------

    st.subheader("Agent System Status")

    col1, col2, col3 = st.columns(3)

    col1.metric("Autoencoder", "ONLINE")
    col2.metric("Isolation Forest", "ONLINE")
    col3.metric("Persistent Storage", "ONLINE")

    st.divider()

    # --------------------------------------------------
    # ASTRONAUT SELECTION
    # --------------------------------------------------

    st.subheader("Mission Monitoring Input")

    subjects = sorted(
        live_df["SUBJECT_ID"].dropna().astype(str).unique()
    )

    if not subjects:
        st.warning("No astronaut observations are available.")
        return

    selected_subject = st.selectbox(
        "Select Astronaut",
        subjects,
        key="agent_subject",
    )

    subject_df = live_df[
        live_df["SUBJECT_ID"].astype(str) == selected_subject
    ].copy()

    subject_df = subject_df.sort_values("MISSION_DAY")

    if subject_df.empty:
        st.warning("No observations found.")
        return

    latest = subject_df.iloc[-1]

    st.write("**Latest ML-Analyzed Observation**")

    a, b, c = st.columns(3)

    a.metric("Mission Day", int(latest["MISSION_DAY"]))
    b.metric(
        "Monitoring Priority",
        str(latest["monitoring_priority"]),
    )
    c.metric(
        "Priority Score",
        int(latest["monitoring_priority_score"]),
    )

    # --------------------------------------------------
    # MODEL OUTPUTS
    # --------------------------------------------------

    st.subheader("Actual Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Isolation Forest**")
        st.write(latest["ISOLATION_FOREST_STATUS"])

        st.metric(
            "Anomaly Score",
            f"{latest['ISOLATION_FOREST_SCORE']:.4f}",
        )

    with col2:
        st.markdown("**PyTorch Autoencoder**")
        st.write(latest["AUTOENCODER_STATUS"])

        st.metric(
            "Reconstruction Error",
            f"{latest['RECONSTRUCTION_ERROR']:.4f}",
        )

    st.caption(
        "Model outputs are experimental anomaly indicators, "
        "not clinical diagnoses."
    )

    st.divider()

    # --------------------------------------------------
    # BACKEND AGENT WORKFLOW
    # --------------------------------------------------

    st.subheader("Agent Execution Workflow")

    st.markdown(
        """
        **Backend agent pipeline**

        1. Data Collection Agent — receives the selected observation.
        2. Anomaly Analysis Agent — reads both model predictions.
        3. Risk Assessment Agent — reads the existing monitoring score.
        4. Recommendation Agent — generates a rule-based recommendation.
        5. Human Review — records an approval or rejection in SQLite.
        """
    )

    if st.button(
        "EXECUTE AGENT WORKFLOW",
        use_container_width=True,
        type="primary",
    ):

        try:
            # Convert the selected live observation into a backend record.
            agent_result = orchestrate_monitoring(latest.to_dict())

            # Save the result permanently in SQLite.
            execution_id = save_agent_execution(agent_result)

            st.session_state.agent_result = agent_result
            st.session_state.agent_execution_id = execution_id

            approval_status = (
                "Pending"
                if agent_result["approval_required"]
                else "Not Required"
            )

            st.session_state.agent_approval = approval_status

            for step in agent_result["workflow"]:
                st.session_state.agent_logs.append({
                    "Agent": step["agent"],
                    "Action": "Monitoring workflow execution",
                    "Status": step["status"],
                    "Astronaut": agent_result["subject_id"],
                    "Sample": agent_result["sample_name"],
                })

            st.success(
                f"Agent workflow completed and saved. "
                f"Execution ID: {execution_id}"
            )

        except Exception as error:
            st.error(f"Agent execution failed: {error}")

    # --------------------------------------------------
    # AGENT RESULT
    # --------------------------------------------------

    result = st.session_state.agent_result

    if result is not None:

        st.divider()
        st.subheader("Agent Analysis Result")

        st.write(f"**Execution ID:** {st.session_state.agent_execution_id}")
        st.write(f"**Astronaut:** {result['subject_id']}")
        st.write(f"**Sample:** {result['sample_name']}")
        st.write(f"**Observation:** {result['timepoint']}")
        st.write(f"**Monitoring Priority:** {result['monitoring_priority']}")
        st.write(
            f"**Experimental Score:** "
            f"{result['monitoring_priority_score']}/80"
        )

        st.markdown("### Generated Recommendation")
        st.success(result["recommendation"])

        st.markdown("### Model Explanation")

        if result["explanations"]:
            for explanation in result["explanations"]:
                st.write(f"- {explanation}")
        else:
            st.write("No additional model explanation was provided.")

        st.markdown("### Workflow Status")

        workflow_df = pd.DataFrame(result["workflow"])

        st.dataframe(
            workflow_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(result["disclaimer"])

        # --------------------------------------------------
        # HUMAN APPROVAL
        # --------------------------------------------------

        st.divider()
        st.subheader("Human Approval Checkpoint")

        st.warning(
            "Review the recommendation before recording a decision. "
            "Approval does not trigger any spacecraft or medical action."
        )

        execution_id = st.session_state.agent_execution_id
        approval_status = st.session_state.agent_approval

        st.write(f"**Current approval status:** {approval_status}")

        if result["approval_required"] and approval_status == "Pending":

            reviewer_note = st.text_area(
                "Reviewer note (optional)",
                key=f"review_note_{execution_id}",
                placeholder="Enter a short review note...",
            )

            approve_col, reject_col = st.columns(2)

            with approve_col:
                if st.button(
                    "APPROVE RECOMMENDATION",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        updated = update_approval(
                            execution_id,
                            "approved",
                            reviewer_note,
                        )

                        if updated:
                            st.session_state.agent_approval = "Approved"
                            st.session_state.agent_logs.append({
                                "Agent": "Human Review",
                                "Action": "Recommendation approved",
                                "Status": "Approved",
                                "Astronaut": result["subject_id"],
                                "Sample": result["sample_name"],
                            })
                            st.success("Approval saved to database.")
                            st.rerun()
                        else:
                            st.error("Execution record was not found.")

                    except Exception as error:
                        st.error(f"Could not save approval: {error}")

            with reject_col:
                if st.button(
                    "REJECT RECOMMENDATION",
                    use_container_width=True,
                ):
                    try:
                        updated = update_approval(
                            execution_id,
                            "rejected",
                            reviewer_note,
                        )

                        if updated:
                            st.session_state.agent_approval = "Rejected"
                            st.session_state.agent_logs.append({
                                "Agent": "Human Review",
                                "Action": "Recommendation rejected",
                                "Status": "Rejected",
                                "Astronaut": result["subject_id"],
                                "Sample": result["sample_name"],
                            })
                            st.warning("Rejection saved to database.")
                            st.rerun()
                        else:
                            st.error("Execution record was not found.")

                    except Exception as error:
                        st.error(f"Could not save rejection: {error}")

        elif result["approval_required"]:
            st.info(
                f"Human decision recorded: {approval_status}"
            )

        else:
            st.success(
                "Routine monitoring: no approval is required "
                "by the current workflow rules."
            )

    # --------------------------------------------------
    # CURRENT SESSION LOGS
    # --------------------------------------------------

    st.divider()
    st.subheader("Current Session Agent Logs")

    if st.session_state.agent_logs:

        logs_df = pd.DataFrame(st.session_state.agent_logs)

        st.dataframe(
            logs_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Current Session Logs",
            data=logs_df.to_csv(index=False).encode("utf-8"),
            file_name="astrahealth_agent_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.info("No agent executions in this session yet.")

    # --------------------------------------------------
    # PERSISTENT EXECUTION HISTORY
    # --------------------------------------------------

    st.divider()
    st.subheader("Persistent Agent Execution History")

    try:
        history = get_recent_executions(limit=20)

        if history:
            history_df = pd.DataFrame([
                {
                    "Execution ID": item["id"],
                    "Astronaut": item["subject_id"],
                    "Sample": item["sample_name"],
                    "Priority": item["priority"],
                    "Score": item["priority_score"],
                    "Approval": item["approval_status"],
                    "Created At": item["created_at"],
                    "Updated At": item["updated_at"],
                }
                for item in history
            ])

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Persistent Execution History",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="astrahealth_execution_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.info("No saved agent executions found.")

    except Exception as error:
        st.error(f"Could not load execution history: {error}")

    st.caption(
        "ASTRAHEALTH AI | ML-powered astronaut health research prototype"
    )
