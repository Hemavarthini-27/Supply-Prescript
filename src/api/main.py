from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import pulp
from datetime import datetime
import os
from database.database import (
    create_tables,
    save_decision,
    save_feedback,
    get_connection
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from database.database import save_training_history

app = FastAPI(title="Supply Prescript API")
create_tables()
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

    # Objective: minimize cost
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

    # Solve optimization
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Find selected action
    selected_action = None

    for action, variable in decision_vars.items():
        if variable.value() == 1:
            selected_action = action
            break

    # Evaluate all alternatives against constraints
    alternatives = []

    for action, details in actions.items():

        budget_ok = details["cost"] <= scenario["available_budget"]
        delay_ok = details["delay_days"] <= scenario["maximum_acceptable_delay"]
        capacity_ok = details["capacity"] >= scenario["required_units"]

        feasible = budget_ok and delay_ok and capacity_ok

        alternatives.append({
            "action": action,
            "cost": details["cost"],
            "delay_days": details["delay_days"],
            "capacity": details["capacity"],
            "feasible": feasible,
            "budget_ok": budget_ok,
            "delay_ok": delay_ok,
            "capacity_ok": capacity_ok,
            "recommended": action == selected_action
        })

    return {
        "selected_action": selected_action,
        "selected_cost": actions[selected_action]["cost"],
        "selected_delay_days": actions[selected_action]["delay_days"],
        "selected_capacity": actions[selected_action]["capacity"],
        "alternatives": alternatives,
        "constraints": {
            "required_units": scenario["required_units"],
            "available_budget": scenario["available_budget"],
            "maximum_acceptable_delay": scenario["maximum_acceptable_delay"]
        }
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

@app.post("/evaluate-decision")
def evaluate_decision(data: dict):

    decision_id = data["decision_id"]
    actual_delay = data["actual_delay"]
    actual_cost = data["actual_cost"]

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            shipment_id,
            predicted_delay,
            predicted_cost
        FROM executed_decisions
        WHERE id = ?
    """, (decision_id,))

    decision = cursor.fetchone()
    conn.close()

    if not decision:
        return {
            "error": "Decision not found"
        }

    shipment_id = decision[0]
    predicted_delay = decision[1]
    predicted_cost = decision[2]

    feedback = save_feedback(
        decision_id=decision_id,
        shipment_id=shipment_id,
        predicted_delay=predicted_delay,
        actual_delay=actual_delay,
        predicted_cost=predicted_cost,
        actual_cost=actual_cost
    )

    return {
        "message": "Decision evaluated successfully",
        "decision_id": decision_id,
        "shipment_id": shipment_id,
        "predicted_delay": predicted_delay,
        "actual_delay": actual_delay,
        "predicted_cost": predicted_cost,
        "actual_cost": actual_cost,
        "outcome": feedback["outcome"],
        "delay_difference": feedback["delay_difference"],
        "cost_difference": feedback["cost_difference"]
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
        # Save decision to SQLite database
    decision_id = save_decision(
    shipment_id=data["Shipment_ID"],
    predicted_delay=data["Predicted_Delay"],
    recommended_action=data["Recommended_Action"],
    predicted_cost=data["Predicted_Cost"]
)

    return {
    "message": "Decision executed and saved successfully!",
    "decision_id": decision_id,
    "decision": decision_record
}
@app.get("/decision-analytics")
def decision_analytics():

    conn = get_connection()
    cursor = conn.cursor()

    # Total number of decisions
    cursor.execute("""
        SELECT COUNT(*)
        FROM executed_decisions
    """)
    total_decisions = cursor.fetchone()[0]

    # Total estimated cost
    cursor.execute("""
        SELECT COALESCE(SUM(predicted_cost), 0)
        FROM executed_decisions
    """)
    total_cost = cursor.fetchone()[0]

    # Average predicted delay
    cursor.execute("""
        SELECT COALESCE(AVG(predicted_delay), 0)
        FROM executed_decisions
    """)
    average_delay = cursor.fetchone()[0]

    # Most frequently recommended action
    cursor.execute("""
        SELECT recommended_action, COUNT(*) AS action_count
        FROM executed_decisions
        GROUP BY recommended_action
        ORDER BY action_count DESC
        LIMIT 1
    """)
    result = cursor.fetchone()

    most_recommended_action = result[0] if result else "No decisions yet"

    conn.close()

    return {
        "total_decisions": total_decisions,
        "total_cost": round(float(total_cost), 2),
        "average_predicted_delay": round(float(average_delay), 2),
        "most_recommended_action": most_recommended_action
    }
    
@app.post("/retrain-model")
def retrain_model():

    # Load dataset
    dataset = pd.read_csv("data/processed/cleaned_supply_chain.csv")

    # Target column
    y = dataset["Late_delivery_risk"]

    # Features
    X = dataset.drop(columns=["Late_delivery_risk"])

    # Transform using existing preprocessor
    X_processed = preprocessor.transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train updated model
    new_model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )

    new_model.fit(X_train, y_train)

    # Evaluate accuracy
    predictions = new_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Save updated model
    joblib.dump(
        new_model,
        "models/xgboost_delay_model.pkl"
    )

    # Save training history
    save_training_history(
        model_version="v1.1",
        accuracy=round(accuracy * 100, 2),
        trigger_reason="Closed-loop feedback"
    )

    return {
        "message": "Model retrained successfully!",
        "accuracy": round(accuracy * 100, 2),
        "model_version": "v1.1"
    }

@app.post("/evaluate-decision")
def evaluate_decision(data: dict):

    decision_id = data["decision_id"]
    shipment_id = data["shipment_id"]

    predicted_delay = data["predicted_delay"]
    actual_delay = data["actual_delay"]

    predicted_cost = data["predicted_cost"]
    actual_cost = data["actual_cost"]

    feedback = save_feedback(
        decision_id=decision_id,
        shipment_id=shipment_id,
        predicted_delay=predicted_delay,
        actual_delay=actual_delay,
        predicted_cost=predicted_cost,
        actual_cost=actual_cost
    )

    return {
        "message": "Decision evaluated successfully!",
        "decision_id": decision_id,
        "shipment_id": shipment_id,
        "predicted_delay": predicted_delay,
        "actual_delay": actual_delay,
        "delay_difference": feedback["delay_difference"],
        "predicted_cost": predicted_cost,
        "actual_cost": actual_cost,
        "cost_difference": feedback["cost_difference"],
        "outcome": feedback["outcome"]
    }
    
@app.get("/decision-history")
def decision_history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            shipment_id,
            predicted_delay,
            recommended_action,
            predicted_cost,
            execution_date
        FROM executed_decisions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    history = []

    for row in rows:
        history.append({
            "decision_id": row[0],
            "shipment_id": row[1],
            "predicted_delay": row[2],
            "recommended_action": row[3],
            "predicted_cost": row[4],
            "execution_date": row[5]
        })

    return {
        "total_decisions": len(history),
        "decisions": history
    }

    
@app.get("/feedback-analytics")
def feedback_analytics():

    conn = get_connection()
    cursor = conn.cursor()

    # Total evaluated decisions
    cursor.execute("""
        SELECT COUNT(*)
        FROM decision_feedback
    """)
    total_evaluated = cursor.fetchone()[0]

    # Successful decisions
    cursor.execute("""
        SELECT COUNT(*)
        FROM decision_feedback
        WHERE outcome = 'Successful'
    """)
    successful = cursor.fetchone()[0]

    # Partially successful decisions
    cursor.execute("""
        SELECT COUNT(*)
        FROM decision_feedback
        WHERE outcome = 'Partially Successful'
    """)
    partially_successful = cursor.fetchone()[0]

    # Unsuccessful decisions
    cursor.execute("""
        SELECT COUNT(*)
        FROM decision_feedback
        WHERE outcome = 'Unsuccessful'
    """)
    unsuccessful = cursor.fetchone()[0]

    # Average cost difference
    cursor.execute("""
        SELECT COALESCE(AVG(cost_difference), 0)
        FROM decision_feedback
    """)
    average_cost_difference = cursor.fetchone()[0]

    # Average delay difference
    cursor.execute("""
        SELECT COALESCE(AVG(delay_difference), 0)
        FROM decision_feedback
    """)
    average_delay_difference = cursor.fetchone()[0]

    # Success rate
    if total_evaluated > 0:
        success_rate = (
            successful / total_evaluated
        ) * 100
    else:
        success_rate = 0

    conn.close()

    return {
        "total_evaluated": total_evaluated,
        "successful": successful,
        "partially_successful": partially_successful,
        "unsuccessful": unsuccessful,
        "success_rate": round(success_rate, 2),
        "average_cost_difference": round(
            float(average_cost_difference), 2
        ),
        "average_delay_difference": round(
            float(average_delay_difference), 2
        )
    }