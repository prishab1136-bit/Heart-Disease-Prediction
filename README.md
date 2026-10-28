# Heart-Disease-Prediction
[project_readme.md](https://github.com/user-attachments/files/34093729/project_readme.md)
# 🫀 Heart Disease Prediction System Using Machine Learning

A clinical decision-support pipeline that leverages supervised machine learning to assess cardiovascular risk. Built using Python, Scikit-Learn, and Pandas, this system evaluates physiological parameters to classify patient records into **Healthy** or **High Cardiac Risk** categories.

---

#Project Overview

Cardiovascular diseases (CVDs) are the leading cause of global mortality. Early, non-invasive diagnostic screening enables proactive intervention and drastically improves patient outcomes. 

This project processes tabular diagnostic data (such as age, blood pressure, serum cholesterol, maximum heart rate, and ST-segment depression) and applies supervised classification algorithms (including **Logistic Regression**, **Random Forest**, **XGBoost**, and **Decision Trees**) to deliver rapid, accurate risk evaluations.

---

#Dataset Schema

The system relies on clinical dataset metrics (e.g., standard UCI Cleveland Heart Disease dataset) containing 13 predictor features and 1 binary output target label:

| Attribute Name | Description | Clinical Units / Range |
| :--- | :--- | :--- |
| `age` | Age of the patient | 29 to 77 years |
| `sex` | Biological sex | `1` = Male, `0` = Female |
| `cp` | Chest pain type | `0`: Typical, `1`: Atypical, `2`: Non-anginal, `3`: Asymptomatic |
| `trestbps` | Resting blood pressure on admission | mmHg (continuous scalar) |
| `chol` | Serum cholesterol level | mg/dL |
| `fbs` | Fasting blood sugar | `1` if > 120 mg/dL, `0` otherwise |
| `restecg` | Resting electrocardiographic results | `0`: Normal, `1`: ST-T wave abnormality, `2`: Hypertrophy |
| `thalach` | Maximum heart rate achieved during stress test | bpm |
| `exang` | Exercise-induced angina | `1` = Yes, `0` = No |
| `oldpeak` | ST depression induced by exercise relative to rest | Float mm |
| `slope` | Slope of the peak exercise ST segment | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | 0 to 4 |
| `thal` | Thalassemia status | `1`: Normal, `2`: Fixed defect, `3`: Reversible defect |
| **`target`** | **Diagnostic Output Flag (Class)** | **`0` = Healthy Heart, `1` = Defective Heart** |

---

#Key Features & Pipeline Architecture

1. **Data Ingestion & Integrity Auditing:**
   - Programmatically parses dataset files and validates spatial dimensions ($303 \times 14$).
   - Scans for missing/null values across all attributes.

2. **Feature Engineering & Preprocessing:**
   - Standardizes continuous physiological metrics using `StandardScaler` ($Z = \frac{x - \mu}{\sigma}$).
   - Categorical feature encoding and class distribution verification.

3. **Stratified Train-Test Splitting:**
   - Partitions data into **80% Training** ($242$ samples) and **20% Testing** ($61$ samples) subsets using stratified sampling to prevent class imbalance bias.

4. **Model Training & Hyperparameter Tuning:**
   - Implements supervised classification models using solvers like L-BFGS and ensemble tree aggregation.

5. **Interactive Predictive Engine:**
   - Built-in single-instance inference routine capable of taking new patient physiological tuples and returning real-time risk predictions.

---

#Model Performance & Results

Under an 80:20 stratified split, model benchmarks demonstrate high clinical precision and sensitivity:

| Algorithm / Classifier | Training Accuracy | Testing Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 100.0% | **90.16%** | 89.29% | **90.62%** | 89.95% | **0.933** |
| **XGBoost Classifier** | 98.2% | **88.52%** | 87.50% | 89.10% | 88.29% | 0.915 |
| **Logistic Regression** | 85.12% | **81.97%** | 83.87% | 86.66% | 85.24% | 0.892 |
| **Decision Tree** | 100.0% | **78.68%** | 76.47% | 81.25% | 78.79% | 0.785 |

> **Note:** High **Recall (Sensitivity)** is prioritized in clinical settings to minimize false negatives and ensure at-risk patients receive medical intervention.

---

#Tech Stack & Dependencies

- **Language:** Python 3.8+
- **Environment:** Jupyter Notebook / Google Colab
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning Engine:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`

---

#Quick Start & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3. Run Notebook or Python Execution
Launch Jupyter Notebook or execute the Python script:
```bash
jupyter notebook Heart_Disease_Prediction.ipynb
```

### 4. Interactive Prediction Example
```python
import numpy as np

# Sample Patient Input: (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_patient_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# Reshape array for single sample inference
input_data_reshaped = np.asarray(input_patient_data).reshape(1, -1)

# Generate Prediction
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("Diagnosis: Patient is HEALTHY (Low Risk)")
else:
    print("Diagnosis: Patient HAS Heart Disease (High Risk - Medical Evaluation Advised)")
```

---

#Future Enhancements

- **Web & Mobile Dashboard:** Deploying interactive clinician UIs using Streamlit or FastAPI + React.
- **Explainable AI (XAI):** Integrating **SHAP** and **LIME** to give doctors transparent feature-importance breakdowns for each patient.
- **Wearable IoT Feeds:** Streaming real-time heart rate and ECG data from smart wearables.

---

#Author & Acknowledgements

- **Author:** Prisha Biswas
- **Department:** Department of B.Pharm
- **Institution:** JSPM Rajarshi Shahu College of Pharmacy & Research, Pune
- **Academic Session:** 2026–2027
