from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import pulp
from datetime import datetime
import os

app = FastAPI(title="Supply Prescript API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model and preprocessor
model = joblib.load("models/xgboost_delay_model.pkl")
preprocessor = joblib.load("models/xgboost_preprocessor.pkl")


@app.get("/")
def home():
    return {
        "message": "Supply Prescript API is running successfully!"
    }

def optimize_supply_chain():

    actions = {
        "Air Freight": {
            "cost": 15000,
            "delay_days": 2,
            "capacity": 150
        },
        "Secondary Supplier": {
            "cost": 12500,
            "delay_days": 5,
            "capacity": 120
        },
        "Delay Product Launch": {
            "cost": 5000,
            "delay_days": 14,
            "capacity": 100
        }
    }

    scenario = {
        "required_units": 100,
        "available_budget": 20000,
        "maximum_acceptable_delay": 7
    }

    # Create optimization problem
    problem = pulp.LpProblem(
        "Supply_Prescript_Optimization",
        pulp.LpMinimize
    )

    # Binary decision variables
    decision_vars = {
        action: pulp.LpVariable(
            action.replace(" ", "_"),
            cat="Binary"
        )
        for action in actions
    }

    # Minimize cost
    problem += pulp.lpSum(
        actions[action]["cost"] * decision_vars[action]
        for action in actions
    )

    # Choose exactly one action
    problem += pulp.lpSum(
        decision_vars[action]
        for action in actions
    ) == 1

    # Budget constraint
    problem += pulp.lpSum(
        actions[action]["cost"] * decision_vars[action]
        for action in actions
    ) <= scenario["available_budget"]

    # Delay constraint
    problem += pulp.lpSum(
        actions[action]["delay_days"] * decision_vars[action]
        for action in actions
    ) <= scenario["maximum_acceptable_delay"]

    # Capacity constraint
    problem += pulp.lpSum(
        actions[action]["capacity"] * decision_vars[action]
        for action in actions
    ) >= scenario["required_units"]

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Identify selected action
    selected_action = None

    for action, variable in decision_vars.items():
        if variable.value() == 1:
            selected_action = action
            break

    return {
        "action": selected_action,
        "cost": actions[selected_action]["cost"],
        "delay_days": actions[selected_action]["delay_days"],
        "capacity": actions[selected_action]["capacity"]
    }
    
@app.post("/predict")
def predict_delay(data: dict):

    # Convert incoming JSON into DataFrame
    input_df = pd.DataFrame([data])

    # Preprocess input
    processed_data = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(processed_data)[0]

    # Probability of delay
    probability = model.predict_proba(processed_data)[0][1]

    # Risk and recommendation
    if probability < 0.30:
        risk_level = "LOW RISK"
        recommendation = "Proceed with current supplier"

    elif probability < 0.60:
        risk_level = "MEDIUM RISK"
        recommendation = "Monitor shipment and prepare backup supplier"

    else:
        risk_level = "HIGH RISK"
        recommendation = "Activate backup supplier / expedite shipment"

    # Prescriptive optimization
    optimization = optimize_supply_chain()

    return {
        "prediction": int(prediction),
        "prediction_label": "Delayed" if prediction == 1 else "On Time",
        "delay_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "optimization": optimization
    }

@app.post("/execute-decision")
def execute_decision(data: dict):

    decision_record = {
        "Shipment_ID": data["Shipment_ID"],
        "Predicted_Delay": data["Predicted_Delay"],
        "Recommended_Action": data["Recommended_Action"],
        "Predicted_Cost": data["Predicted_Cost"],
        "Execution_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    file_path = "data/processed/executed_decisions.csv"

    decision_df = pd.DataFrame([decision_record])

    if os.path.exists(file_path):
        decision_df.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        decision_df.to_csv(
            file_path,
            index=False
        )

    return {
        "message": "Decision executed and saved successfully!",
        "decision": decision_record
    }