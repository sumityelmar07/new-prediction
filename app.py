import streamlit as st
st.set_page_config(page_title="Cricket Match Outcome Predictor", layout="wide")

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

page_bg_css = """
<style>
body {
    background: linear-gradient(135deg, #a1c4fd, #c2e9fb);
    background-image: url("https://th.bing.com/th/id/OIP.jrtujbLuXMWQVWbRExVwvgHaEC?w=1833&h=1001&rs=1&pid=ImgDetMain");
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv("cricket_dataset.csv")
    return data

@st.cache_resource
def train_model(data):
    features = [
        "team1", "team2", "toss_winner", "toss_decision", "home_team",
        "crowd_support", "pitch_condition", "weather_condition",
        "team1_strength", "team2_strength", "team1_recent_form", "team2_recent_form", "match_importance"
    ]
    target = "winner"
    X = data[features]
    y = data[target]

    categorical_cols = [
        "team1", "team2", "toss_winner", "toss_decision", "home_team",
        "weather_condition", "match_importance"
    ]
    numerical_cols = [
        "crowd_support", "pitch_condition", "team1_strength",
        "team2_strength", "team1_recent_form", "team2_recent_form"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10]
    }
    grid_search = GridSearchCV(
        clf_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search, X_train, X_test, y_train, y_test

st.title("Cricket Match Outcome Predictor")
st.markdown(
    """
    This app predicts the winner of a cricket match based on pre-match details.
    Adjust the inputs on the sidebar to see the prediction update in real-time!
    """
)

data = load_data()

st.sidebar.header("Dataset & Model Info")
if st.sidebar.checkbox("Show Raw Dataset"):
    st.sidebar.write(data.head())

model, grid_search, X_train, X_test, y_train, y_test = train_model(data)
st.sidebar.markdown("### Best Model Parameters")
st.sidebar.write(grid_search.best_params_)

if st.sidebar.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

st.markdown("## Enter Match Details for Prediction")
team_options = sorted(list(set(data["team1"].unique().tolist() + data["team2"].unique().tolist())))

st.sidebar.header("Match Input Parameters")
team1_input = st.sidebar.selectbox("Select Team 1", team_options)
team2_input = st.sidebar.selectbox("Select Team 2 (different from Team 1)", [team for team in team_options if team != team1_input])
toss_winner_input = st.sidebar.selectbox("Select Toss Winner", [team1_input, team2_input])
toss_decision_input = st.sidebar.selectbox("Toss Decision", ["bat", "bowl"])
home_team_input = st.sidebar.selectbox("Home Team", [team1_input, team2_input])
crowd_support_input = st.sidebar.number_input("Crowd Support (spectators)", min_value=10000, max_value=50000, value=30000, step=1000)
pitch_condition_input = st.sidebar.slider("Pitch Condition (1-10)", min_value=1, max_value=10, value=8)
weather_condition_input = st.sidebar.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy"])
team1_strength_input = st.sidebar.slider("Team 1 Strength (50-100)", min_value=50, max_value=100, value=85)
team2_strength_input = st.sidebar.slider("Team 2 Strength (50-100)", min_value=50, max_value=100, value=80)
team1_recent_form_input = st.sidebar.slider("Team 1 Recent Form (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
team2_recent_form_input = st.sidebar.slider("Team 2 Recent Form (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
match_importance_input = st.sidebar.selectbox("Match Importance", ["Group Stage", "Knockout", "Final"])

user_input = {
    "team1": team1_input,
    "team2": team2_input,
    "toss_winner": toss_winner_input,
    "toss_decision": toss_decision_input,
    "home_team": home_team_input,
    "crowd_support": crowd_support_input,
    "pitch_condition": pitch_condition_input,
    "weather_condition": weather_condition_input,
    "team1_strength": team1_strength_input,
    "team2_strength": team2_strength_input,
    "team1_recent_form": team1_recent_form_input,
    "team2_recent_form": team2_recent_form_input,
    "match_importance": match_importance_input
}
user_input_df = pd.DataFrame([user_input])

if hasattr(model.named_steps["classifier"], "predict_proba"):
    proba = model.predict_proba(user_input_df)[0]
    classes = model.named_steps["classifier"].classes_
    
    p1 = proba[list(classes).index(team1_input)] if team1_input in classes else 0
    p2 = proba[list(classes).index(team2_input)] if team2_input in classes else 0
    
    if (p1 + p2) > 0:
        normalized_team1 = int(round((p1 / (p1 + p2)) * 100))
        normalized_team2 = 100 - normalized_team1
    else:
        normalized_team1 = 0
        normalized_team2 = 0

    st.markdown("### Win Probabilities for Each Team")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{team1_input} Win Probability**")
        progress_bar_team1 = st.progress(0)
        for i in range(normalized_team1 + 1):
            time.sleep(0.005)  
            progress_bar_team1.progress(i)
        st.write(f"Probability: {normalized_team1}%")
        
    with col2:
        st.write(f"**{team2_input} Win Probability**")
        progress_bar_team2 = st.progress(0)
        for i in range(normalized_team2 + 1):
            time.sleep(0.005)  
            progress_bar_team2.progress(i)
        st.write(f"Probability: {normalized_team2}%")

predicted_winner = model.predict(user_input_df)[0]
st.markdown("## Final Prediction Result")
st.write("### Predicted Winner:", predicted_winner)

st.markdown("### How It Works")
st.markdown(
    """
    1. **Data Loading & Model Training:**  
       The app loads a synthetic cricket dataset and trains a RandomForest classifier with hyperparameter tuning using GridSearchCV.
       
    2. **Interactive Prediction:**  
       Adjust match details using the sidebar. The app calculates win probabilities for each team, normalizes them to sum to 100%, and displays them with progress bars.
       
    3. **Final Prediction:**  
       Based on the input parameters, the app predicts the match winner.
    """
)
