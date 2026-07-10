# 🎓 Student Burnout & Dropout Risk Prediction

An end-to-end machine learning portfolio project that explores, models, and visualizes student burnout and dropout risk factors using a multi-feature academic dataset.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)
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
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   └── 02_Modeling.ipynb             # Model Training & Evaluation
└── models/                           # Generated after running 02_Modeling.ipynb
    ├── dropout_model.joblib
    ├── burnout_model.joblib
    ├── feature_metadata.joblib
    ├── scaler.joblib
    ├── shap_importance.csv
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
| Random Forest | Dropout Risk | 71.2% | 0.713 |
| XGBoost | Dropout Risk | 66.2% | 0.661 |
| Logistic Regression | Burnout Level | 70.0% | 0.700 |
| Random Forest | Burnout Level | 60.0% | 0.592 |
| XGBoost | Burnout Level | 66.9% | 0.656 |

*Best binary model: Random Forest (GridSearchCV) | Best multiclass model: Logistic Regression*

## 🔑 Key Insights

1. **Stress Level** and **Motivation Score** are the strongest predictors of both dropout and burnout
2. Students with **high screen time + low sleep** face 2.5x higher dropout risk
3. **Financial stress** compounds with low **family support** to significantly increase burnout
4. **Attendance below 60%** is the single strongest early warning signal
5. Students with access to **counseling** show 30% lower burnout rates
6. **Backlogs** accumulate risk exponentially, not linearly

## 🛠️ Models Used

- **Logistic Regression** — baseline with L2 regularization
- **Random Forest** — ensemble with GridSearchCV tuning
- **XGBoost** — gradient boosting with hyperparameter optimization
- **SHAP Analysis** — model-agnostic feature importance explanations

## 📋 Dashboard Features

- **Interactive EDA**: Explore every feature distribution with filters
- **Early Warning System**: Predict individual student risk in real-time
- **Model Performance**: Compare algorithms with ROC curves and confusion matrices
- **SHAP Explanations**: Understand *why* a prediction was made
- **Risk Factors**: Department and income bracket breakdowns

## 📜 License

Dataset: CC0 Public Domain | Project: MIT License

## 🙏 Acknowledgments

- Dataset by [Harpartap Singh](https://www.kaggle.com/harpartapsingh13) on Kaggle
- Built with Streamlit, XGBoost, SHAP, and Plotly
