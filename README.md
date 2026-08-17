# Supply Prescript

## AI-Powered Supply Chain Delay Prediction & Prescriptive Decision Support System

Supply Prescript is an end-to-end machine learning and analytics application designed to predict shipment delays, assess operational risk, recommend corrective actions, and evaluate the effectiveness of executed decisions through a closed-loop feedback system.

## 🚀 Key Features

* **Shipment Delay Prediction**

  * Predicts whether a shipment is likely to be delayed or on time.
  * Calculates delay probability.

* **Risk Assessment**

  * Categorizes shipments into:

    * Low Risk
    * Medium Risk
    * High Risk

* **Prescriptive Optimization**

  * Recommends an appropriate operational action.
  * Compares alternative actions based on:

    * Estimated cost
    * Expected delay
    * Available capacity
    * Feasibility

* **Decision Execution**

  * Executes the recommended decision.
  * Stores executed decisions in SQLite and processed decision data.

* **Decision Analytics**

  * Tracks total decisions.
  * Calculates total cost.
  * Calculates average predicted delay.
  * Identifies the most frequently recommended action.

* **Closed-Loop Performance Evaluation**

  * Compares predicted outcomes with actual outcomes.
  * Tracks:

    * Successful decisions
    * Partially successful decisions
    * Unsuccessful decisions
    * Success rate
    * Average cost difference
    * Average delay difference

* **Model Retraining**

  * Provides a retraining endpoint.
  * Records model version and accuracy in training history.

* **Decision History**

  * Displays previously executed decisions with their prediction, recommended action, cost, and execution date.

---

## 🏗️ System Workflow

```text
Enter Shipment Details
        ↓
Predict Shipment Delay
        ↓
Calculate Delay Probability
        ↓
Determine Risk Level
        ↓
Generate Recommended Action
        ↓
Prescriptive Optimization
        ↓
Execute Recommended Decision
        ↓
Store Decision
        ↓
Evaluate Actual Outcome
        ↓
Closed-Loop Performance Analytics
        ↓
Retrain Model
```

---

## 🧠 Machine Learning

The project uses trained machine learning models for shipment-related prediction and delay analysis.

The prediction pipeline processes both categorical and numerical shipment features.

### Categorical Features

* Shipping Mode
* Market
* Order Region
* Category Name
* Customer Segment
* Department Name
* Order Weekday
* Order Country
* Order State

### Numerical Features

* Days for shipment (scheduled)
* Order Item Quantity
* Sales
* Product Price
* Order Year
* Order Month
* Order Day

The project uses preprocessing and trained model artifacts stored in the `models` directory.

---

## 🛠️ Technology Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib

### Database

* SQLite

### Development

* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
Supply-Prescript/
│
├── dashboard/
│   └── index.html
│
├── src/
│   └── api/
│       └── main.py
│
├── database/
│   ├── database.py
│   └── supply_prescript.db
│
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_delay_model.pkl
│   └── label_encoder.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hemavarthini-27/Supply-Prescript.git
cd Supply-Prescript
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI backend

From the project root:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the dashboard

Open the `dashboard/index.html` file using a local development server.

The dashboard communicates with the FastAPI backend through the API endpoints.

---

## 🔌 API Endpoints

| Method | Endpoint              | Purpose                                 |
| ------ | --------------------- | --------------------------------------- |
| `GET`  | `/`                   | API status                              |
| `POST` | `/predict`            | Predict shipment delay                  |
| `POST` | `/execute-decision`   | Execute and save a recommended decision |
| `GET`  | `/decision-analytics` | Retrieve decision analytics             |
| `GET`  | `/feedback-analytics` | Retrieve closed-loop performance        |
| `POST` | `/evaluate-decision`  | Evaluate an executed decision           |
| `GET`  | `/decision-history`   | Retrieve executed decision history      |
| `POST` | `/retrain-model`      | Retrain the model                       |

Swagger UI can be used to test the APIs individually.

---

## 🔄 Closed-Loop Decision System

The closed-loop component compares the model's predictions and recommended decisions with actual shipment outcomes.

For each evaluated decision, the system records:

```text
Decision ID
Shipment ID
Predicted Delay
Actual Delay
Predicted Cost
Actual Cost
Delay Difference
Cost Difference
Outcome
```

The outcome is classified as:

* **Successful**
* **Partially Successful**
* **Unsuccessful**

This feedback is then summarized through the dashboard's Decision Analytics section.

---

## 📊 Decision Analytics

The dashboard provides operational metrics including:

* Total decisions
* Total cost
* Average predicted delay
* Most recommended action
* Total evaluated decisions
* Successful decisions
* Partially successful decisions
* Unsuccessful decisions
* Success rate
* Average cost difference
* Average delay difference

---

## 🤖 Model Retraining

The system provides a model retraining endpoint:

```text
POST /retrain-model
```

The retraining process returns information such as:

```text
Model Version
Accuracy
Training Status
```

Training information is stored in the `model_training_history` table for tracking model versions over time.

---

## 💾 Database

SQLite is used for persistent storage.

The database stores:

### `executed_decisions`

Stores executed operational decisions.

### `decision_feedback`

Stores actual outcomes and evaluates decision performance.

### `model_training_history`

Stores model versions, accuracy, retraining trigger information, and training dates.

---

## 🧪 Testing

The complete application workflow has been tested through:

1. Shipment input
2. Prediction
3. Risk assessment
4. Recommendation
5. Prescriptive optimization
6. Decision execution
7. Decision ID generation
8. Decision evaluation
9. Feedback analytics
10. Decision history
11. Model retraining

The prediction system was also tested with different shipment input values to verify that prediction probability and risk level can change according to the supplied inputs.

---

## 🎯 Project Objective

The goal of Supply Prescript is to move beyond simply predicting shipment delays.

Instead, the system follows a complete decision-support approach:

```text
Predict → Assess → Recommend → Optimize → Execute → Evaluate → Learn
```

This enables the system to connect machine learning predictions with operational decision-making and feedback-based performance evaluation.

---

## 👩‍💻 Author

**S. Hemavarthini**

B.Tech — Artificial Intelligence and Data Science

GitHub: `Hemavarthini-27`
