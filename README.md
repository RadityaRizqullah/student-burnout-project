# 🎓 Student Burnout & Dropout Risk Prediction

An end-to-end machine learning portfolio project that explores, models, and visualizes student burnout and dropout risk factors using a multi-feature academic dataset.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.12+-yellow?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge)

## 📊 Dataset

800 synthetic student records combining:
- **Demographics**: Age, Gender, Year, Department, Residence
- **Academic**: Attendance, Study Hours, GPA, Backlogs
- **Lifestyle**: Sleep, Screen Time, Exercise, Social Activity
- **Financial**: Income Bracket, Financial Stress, Family Support
- **Mental Health**: Stress, Anxiety, Motivation, Peer Pressure, Counseling

**Targets**: `Dropout_Risk` (Yes/No) and `Burnout_Level` (Low/Medium/High)

## 🏗️ Project Structure

```
student-burnout-project/
├── data.csv                          # Raw dataset
├── requirements.txt                  # Python dependencies
├── app.py                            # Streamlit dashboard
├── README.md                         # This file
├── PROJECT.md                        # Detailed project documentation
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   └── 02_Modeling.ipynb             # Model Training & Evaluation
└── models/                           # Generated after running 02_Modeling.ipynb
    ├── dropout_model.joblib
    ├── burnout_model.joblib
    ├── feature_metadata.joblib
    ├── scaler.joblib
    ├── shap_importance.csv
    ├── model_metrics.json
    └── *.png                         # Plots (ROC, SHAP, confusion matrices)
```

## 🚀 Quick Start

```bash
# Clone and set up
cd student-burnout-project
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run notebooks in order (02_Modeling generates models, metrics, and plots)
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Modeling.ipynb

# Launch the dashboard (requires models from step above)
streamlit run app.py
```

## 📈 Key Results

| Model | Target | Accuracy | F1-Score |
|-------|--------|----------|----------|
| Logistic Regression | Dropout Risk | 70.0% | 0.701 |
| **Random Forest** | **Dropout Risk** | **71.2%** | **0.713** |
| XGBoost | Dropout Risk | 66.2% | 0.661 |
| **Logistic Regression** | **Burnout Level** | **70.0%** | **0.700** |
| Random Forest | Burnout Level | 60.0% | 0.592 |
| XGBoost | Burnout Level | 66.9% | 0.656 |

*Best binary model: Random Forest (GridSearchCV) | Best multiclass model: Logistic Regression*

## 🔑 Key Insights (from SHAP analysis)

1. **Motivation Score** and **Academic Engagement** are the strongest predictors of dropout risk (mean |SHAP| > 0.64)
2. **Attendance Percent** is the single most actionable early warning signal (mean |SHAP| = 0.518)
3. **Stress Level** and **Anxiety Score** are the top mental-health predictors (mean |SHAP| > 0.40)
4. All 3 engineered features (academic engagement, lifestyle risk, financial burden) rank in the top 10, validating the feature engineering approach
5. Modest model accuracy (60–71%) reflects genuinely weak signal — the strongest feature correlates only 0.24 with the target

## 🛠️ Models Used

- **Logistic Regression** — baseline with L2 regularization, balanced class weights
- **Random Forest** — ensemble with GridSearchCV tuning (n_estimators, max_depth, min_samples)
- **XGBoost** — gradient boosting with hyperparameter optimization (learning_rate, subsample, colsample)
- **SHAP Analysis** — TreeExplainer on best XGBoost model for feature importance explanations

## 📋 Dashboard Features

- **📊 Overview**: Dataset summary, dropout rate, burnout distribution, interactive data table
- **🔍 EDA Explorer**: Select any feature → auto-generated distribution plots by target
- **⚠️ Early Warning System**: Input student attributes → real-time dropout/burnout predictions
- **🤖 Model Performance**: Side-by-side comparison, confusion matrices, SHAP feature importance
- **💡 Insights & Recommendations**: Key findings, department breakdowns, intervention suggestions

## 📜 License

Dataset: CC0 Public Domain | Project: MIT License

## 🙏 Acknowledgments

- Dataset by [Harpartap Singh](https://www.kaggle.com/harpartapsingh13) on Kaggle
- Built with Streamlit, XGBoost, SHAP, and Plotly
