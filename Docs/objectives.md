# ASTRAHEALTH AI — Project Objectives

## Project goal
Develop an experimental astronaut-health monitoring and decision-support prototype that demonstrates how historical physiological measurements can be processed, analyzed for unusual patterns, and presented with explainable monitoring-priority information.

## Implemented objectives
1. **Process astronaut health data:** Use NASA OSDR OSD-569 Inspiration4 whole-blood measurements, perform data-quality review, and prepare a model-ready feature matrix.
2. **Engineer model features:** Use 15 selected CBC measurements and 3 missing-value indicators to create an 18-dimensional model input.
3. **Detect unusual patterns:** Apply Isolation Forest and a PyTorch Autoencoder using reconstruction error as an additional anomaly indicator.
4. **Combine model signals:** Apply an experimental monitoring-priority engine to combine anomaly flags and reconstruction-error conditions.
5. **Present monitoring information:** Provide a Streamlit dashboard for exploring model outputs, participant observations, trends, and decision-support information.
6. **Structure a review workflow:** Use a rule-based backend monitoring agent to organize an observation, model outputs, and a recommendation.
7. **Persist agent records:** Store agent executions and human approval/rejection decisions in SQLite.
8. **Test software behavior:** Use pytest to test preprocessing, model interfaces, prediction outputs, scoring, agent orchestration, and persistence.

## Research objectives / future work
- Expand the dataset and participant cohort.
- Evaluate additional astronaut-health data sources.
- Investigate participant-specific and temporal modeling.
- Study threshold sensitivity and distribution shift.
- Obtain qualified medical review and define clinically meaningful validation criteria.
- Evaluate operational use with appropriate telemetry, security, reliability, and safety controls.

## Intended users
- Astronaut-health researchers
- Mission-control research teams
- Space-medicine researchers
- Students and developers studying applied machine learning

## Expected demonstration outcome
A working research prototype that loads historical astronaut blood measurements, applies two unsupervised models, generates experimental monitoring-priority outputs, presents those outputs through a dashboard, and records human review decisions.

## Safety and scope
ASTRAHEALTH AI is not a medical device or clinical decision system. It does not diagnose disease, estimate validated medical risk, or replace professional medical judgment. Its anomaly indicators and priority categories are experimental software outputs.
