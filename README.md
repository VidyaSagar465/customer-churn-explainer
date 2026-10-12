#  Customer Churn Risk & Explainability Engine

An interactive end-to-end Machine Learning web application that predicts customer churn probabilities and breaks down individual risk factors using SHAP (SHapley Additive exPlanations).

 **Live Demo:** [https://customer-churn-explainer-ehqcbdltgju3qkhmo99anx.streamlit.app/]

---

##  Project Overview

Customer churn prediction allows businesses to identify high-risk customers before they leave. Standard machine learning models often act as "black boxes"—providing predictions without clear reasoning. 

This project solves that gap by combining **XGBoost** for high-accuracy risk scoring with **SHAP** game-theoretic value attribution to deliver clear, actionable feature-level insights for retention teams.

---

##  Key Features

* **Predictive Risk Scoring:** Trained on 7,000+ IBM customer records using XGBoost (achieving **0.84 ROC-AUC**).
* **Explainable AI (XAI):** Generates real-time SHAP waterfall plots showing exact feature contributions (positive/negative impact) for any specific customer profile.
* **Interactive Scenario Testing:** Built with Streamlit to let non-technical stakeholders dynamically adjust attributes (contract type, tenure, monthly charges) and evaluate risk instantly.

---

##  Tech Stack

* **Language:** Python
* **ML Model:** XGBoost, Scikit-learn
* **Explainability:** SHAP
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Visualization:** Matplotlib
* **Model Persistence:** Joblib

---

##  Repository Structure

```text
├── app.py                   # Streamlit web application dashboard
├── churn_notebook.ipynb     # Jupyter Notebook (EDA, preprocessing, XGBoost, SHAP)
├── churn_model.pkl          # Trained XGBoost model artifact
├── explainer.pkl            # Persistent SHAP explainer object
├── feature_columns.pkl      # Saved feature structure for input alignment
├── requirements.txt         # Dependencies for deployment
└── README.md                # Project documentation
