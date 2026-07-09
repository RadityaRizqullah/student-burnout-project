# 🎓 Student Burnout & Dropout Risk Prediction

End-to-end machine learning portfolio project that explores, models, and
visualizes factors driving student burnout and academic dropout risk.
Built with Python, scikit-learn, XGBoost, SHAP, and Streamlit.

**Dataset**: 800 synthetic student records, 25 features, 2 targets
**License**: CC0 (Public Domain) — free to use and republish

---

## Project Structure

```
student-burnout-project/
├── data.csv                          # Raw dataset (800 × 25)
├── run_modeling.py                   # Model training pipeline (run this)
├── app.py                            # Streamlit interactive dashboard
├── requirements.txt                  # Python dependencies
├── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis (78 cells)
│   └── 02_Modeling.ipynb             # Modeling walkthrough (47 cells)
│
├── models/                           # Trained artifacts
│   ├── dropout_model.joblib          # Best binary classifier
│   ├── burnout_model.joblib          # Best multiclass classifier
│   ├── feature_metadata.joblib       # Feature names, encoders, model info
│   ├── shap_importance.csv           # SHAP feature rankings
│   └── model_comparison.csv          # All 6 model metrics
│
├── confusion_matrices.png            # 2×3 grid of all confusion matrices
├── model_comparison.png              # Bar chart comparing all model metrics
└── shap_summary.png                  # SHAP beeswarm feature importance plot
```

---

## Quick Start

### Prerequisites

- Python 3.12+ (tested on 3.12.10)
- Windows, macOS, or Linux

### Setup

```bash
cd student-burnout-project

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Model Training Pipeline

This trains 6 models (3 algorithms × 2 targets), runs SHAP analysis,
generates plots, and exports trained models to `models/`:

```bash
python run_modeling.py
```

Takes ~2–3 minutes. Outputs:
- Classification reports for each model
- Confusion matrix heatmaps → `confusion_matrices.png`
- Metrics bar chart → `model_comparison.png`
- SHAP beeswarm plot → `shap_summary.png`
- Trained models → `models/*.joblib`

### Launch the Interactive Dashboard

```bash
streamlit run app.py
```

Opens in browser at `http://localhost:8501` with 5 pages:

| Page | What It Does |
|------|-------------|
| **Overview** | Dataset stats, key metrics, interactive data table |
| **EDA Explorer** | Select any feature → auto-generated distribution by target |
| **Early Warning System** | Input student attributes → get dropout/burnout predictions |
| **Model Performance** | Side-by-side model comparison, confusion matrices, SHAP |
| **Insights & Recommendations** | Key findings, department breakdowns, interventions |

### Open Jupyter Notebooks

```bash
jupyter notebook notebooks/01_EDA.ipynb   # EDA first
jupyter notebook notebooks/02_Modeling.ipynb  # then Modeling
```

---

## Dataset

**Source**: [Kaggle — Harpartap Singh](https://www.kaggle.com/datasets/harpartapsingh13/student-burnout-and-dropout-risk-dataset)

800 student records with 25 columns across 5 categories:

| Category | Columns |
|----------|---------|
| **Demographics** | Age, Gender, Year_of_Study, Department, Residence_Type |
| **Academic** | Attendance_Percent, Study_Hours_Per_Day, Previous_GPA, Backlogs |
| **Lifestyle** | Sleep_Hours, Screen_Time_Hours, Exercise_Freq_Per_Week, Social_Activity_Score, Part_Time_Job |
| **Financial** | Family_Income_Bracket, Financial_Stress_Score, Family_Support_Score |
| **Mental Health** | Stress_Level, Anxiety_Score, Motivation_Score, Peer_Pressure_Score, Counseling_Access |

**Target variables**:
- `Dropout_Risk` — binary (Yes/No) → 437 Yes, 363 No
- `Burnout_Level` — 3-class (Low/Medium/High) → 320 Low, 280 Medium, 200 High

**Missing values**: 612 total across 18 columns — imputed with median (numeric) and mode (categorical).

---

## Modeling Approach

### Preprocessing

1. **Missing values** — median imputation for numeric, mode for categorical
2. **Encoding** — LabelEncoder for all 6 categorical features
3. **Feature engineering** — 3 derived features:
   - `academic_engagement` — weighted combo of attendance, study hours, GPA
   - `lifestyle_risk` — screen time, sleep, exercise composite
   - `financial_burden` — income bracket + financial stress
4. **Scaling** — StandardScaler for Logistic Regression
5. **Split** — 80/20 train/test, stratified by target

### Algorithms

| Model | Binary Config | Multiclass Config |
|-------|--------------|-------------------|
| **Logistic Regression** | L2 penalty, max_iter=1000 | Same, multinomial |
| **Random Forest** | GridSearchCV: n_estimators∈{100,200}, max_depth∈{5,10,None}, min_samples_split∈{2,5} | Same best params |
| **XGBoost** | GridSearchCV: n_estimators∈{100,200}, max_depth∈{3,5,7}, learning_rate∈{0.05,0.1} | Same, mlogloss |

---

## Results

### Model Performance

| Model | Target | Accuracy | F1 | Precision | Recall |
|-------|--------|----------|----|-----------|--------|
| **Logistic Regression** ⭐ | Dropout Risk | 69.4% | 0.693 | 0.693 | 0.694 |
| Random Forest | Dropout Risk | 65.6% | 0.653 | 0.655 | 0.656 |
| XGBoost | Dropout Risk | 66.3% | 0.659 | 0.661 | 0.663 |
| **Logistic Regression** ⭐ | Burnout Level | 75.0% | 0.749 | 0.748 | 0.750 |
| Random Forest | Burnout Level | 65.0% | 0.638 | 0.643 | 0.650 |
| XGBoost | Burnout Level | 67.5% | 0.672 | 0.670 | 0.675 |

⭐ = best model per target

> **Note**: Logistic Regression outperforms more complex models here, likely
> due to the relatively small dataset (800 rows, 25 features). With more data,
> tree-based models would likely gain an edge. The project still demonstrates
> the full ML pipeline — the modeling notebook includes all 3 algorithms with
> hyperparameter tuning for portfolio completeness.

### Top Predictive Features (SHAP)

| Rank | Feature | Mean |SHAP Value| |
|------|---------|------|
| 1 | Motivation Score | 0.574 |
| 2 | Academic Engagement (engineered) | 0.564 |
| 3 | Attendance Percent | 0.512 |
| 4 | Stress Level | 0.398 |
| 5 | Anxiety Score | 0.353 |
| 6 | Screen Time Hours | 0.252 |
| 7 | Lifestyle Risk (engineered) | 0.252 |
| 8 | Financial Burden (engineered) | 0.215 |
| 9 | Previous GPA | 0.199 |
| 10 | Family Support Score | 0.184 |

**Takeaway**: Motivation, engagement, and attendance dominate. The 3 engineered
features all rank in the top 10, validating the feature engineering approach.

---

## Generated Visualizations

| File | Description |
|------|-------------|
| `confusion_matrices.png` | 2×3 grid — all 6 confusion matrices |
| `model_comparison.png` | Bar chart — accuracy, F1, precision, recall across all models |
| `shap_summary.png` | SHAP beeswarm — top 20 features by importance |
| `models/derived_features_dist.png` | Distribution of engineered features |

---

## Dashboard Pages

### 1. Overview
Dataset summary (800 students, 25 features), dropout rate, burnout distribution,
interactive data table with column filtering, and Plotly distribution charts.

### 2. EDA Explorer
Dropdown to select any of the 25 features. Automatically generates distribution
plots colored by Dropout_Risk and Burnout_Level. Includes correlation matrix
and statistical summaries grouped by target.

### 3. Early Warning System
Interactive form with sliders and dropdowns for all 21 student attributes.
Click "Predict Risk" to get:
- Dropout probability (Yes/No with confidence)
- Burnout level prediction (Low/Medium/High with probabilities)
- Key risk factors explanation

### 4. Model Performance
Side-by-side model comparison charts, confusion matrices, SHAP feature importance
plot, and feature ranking table. Shows which algorithm performed best and why.

### 5. Insights & Recommendations
Key findings about burnout and dropout drivers, department-level breakdowns,
risk factor analysis by income bracket, and evidence-based intervention
recommendations.

---

## Requirements

See `requirements.txt` for exact versions. Key dependencies:

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
xgboost >= 1.7
shap >= 0.41
matplotlib >= 3.6
seaborn >= 0.12
plotly >= 5.10
streamlit >= 1.20
joblib >= 1.2
```

---

## Technical Notes

- **Pandas 3.x compatibility**: The pipeline uses `df[col] = df[col].fillna()`
  instead of `fillna(inplace=True)` due to pandas Copy-on-Write behavior.
- **PYTHONPATH contamination**: On some setups, the hermes agent venv injects
  paths into `PYTHONPATH`. Always prefix with `PYTHONPATH=""` when running
  project-specific Python commands.
- **Jupyter kernel**: If the Jupyter kernel dies during notebook execution,
  run `python run_modeling.py` as a standalone script instead — it produces
  the same models and visualizations.

---

## Acknowledgments

- Dataset by [Harpartap Singh](https://www.kaggle.com/harpartapsingh13) on Kaggle
- Built with scikit-learn, XGBoost, SHAP, Streamlit, and Plotly
