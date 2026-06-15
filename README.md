# 🧠 Student Stress Prediction and Monitoring System

An end-to-end Machine Learning project that predicts student stress levels using psychological, academic, social, health, and environmental factors. The system classifies students into **Low**, **Medium**, or **High Stress** categories and provides personalized wellness recommendations through an interactive Streamlit application.

---

## 📌 Project Overview

Student stress is a growing concern that affects academic performance, mental health, and overall well-being. This project leverages Machine Learning techniques to identify stress patterns and provide early insights that can help students, educators, and counselors take preventive action.

The project includes:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Baseline model comparison
- Hyperparameter tuning
- Ensemble learning
- Explainable AI (SHAP)
- Streamlit-based deployment
- Wellness recommendation system

---

## 🎯 Objectives

- Predict student stress levels accurately.
- Compare multiple machine learning algorithms.
- Improve model performance through feature engineering and optimization.
- Provide interpretable predictions using Explainable AI techniques.
- Generate personalized wellness suggestions.
- Deploy the trained model through a user-friendly application.

---

## 📊 Features Used

The model utilizes a combination of student-related factors:

### Psychological Factors
- Anxiety Level
- Depression
- Self-Esteem
- Mental Health History

### Academic Factors
- Study Load
- Academic Performance
- Future Career Concerns

### Social Factors
- Peer Pressure
- Bullying
- Social Support
- Teacher-Student Relationship

### Health Factors
- Sleep Quality
- Headache
- Blood Pressure
- Breathing Problems

### Environmental Factors
- Living Conditions
- Safety
- Noise Level
- Basic Needs

---

## ⚙️ Feature Engineering

To improve predictive performance, several domain-specific features were created:

- Psychological Health Index
- Academic Pressure Index
- Mental Health Burden
- Lifestyle Balance
- Social Pressure Index
- Physical Health Risk
- Environment Quality Index
- Support Gap
- Sleep Risk

All engineered features were created without using the target variable, ensuring a leakage-free machine learning pipeline.

---

## 🤖 Machine Learning Models

### Baseline Models

The following models were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

### Model Improvements

After baseline evaluation:

- Random Forest was optimized using GridSearchCV.
- Cross-validation was performed using Stratified K-Fold.
- A Soft Voting Ensemble was developed.

---

## 🏆 Final Model

The final deployed model is a **Soft Voting Ensemble** consisting of:

- Logistic Regression
- Support Vector Machine (RBF Kernel)
- Tuned Random Forest

The ensemble combines probability outputs from all three models and selects the final prediction using soft voting.

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1 Score
- Confusion Matrix
- Cross Validation

The final model was selected based on overall generalization performance and weighted F1-score.

---

## 🔍 Explainable AI (XAI)

To improve transparency and interpretability, the project includes:

### Feature Importance Analysis
Random Forest feature importance was used to identify the most influential stress-related factors.

### SHAP Explainability
SHAP (SHapley Additive exPlanations) was used to:

- Explain individual predictions
- Interpret feature contributions
- Visualize model behavior
- Improve trust and transparency

---

## 📱 Application Features

The Streamlit application allows users to:

- Enter student-related information
- Predict stress levels instantly
- View prediction confidence scores
- Analyze class probabilities
- Receive wellness recommendations
- Store user responses

---

## 🔄 Project Workflow

```text
Student Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Soft Voting Ensemble Model
      ↓
Stress Prediction
      ↓
Confidence & Probabilities
      ↓
Wellness Recommendations
```

---

## 📂 Repository Structure

```text
Student_Stress_Prediction_and_Monitoring_System/
│
├── notebook/
│   └── MLprojectFULLcode.ipynb
│
├── app/
│   ├── app.py
│   ├── bg.jpg
│   └── responses.csv
│
├── models/
│   ├── final_student_stress_model.pkl
│   └── final_feature_columns.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/krish-hk/Student_Stress_Prediction_and_Monitoring_System.git
cd Student_Stress_Prediction_and_Monitoring_System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

## 📋 Sample Prediction Output

```text
Predicted Stress Level: High Stress

Confidence: 91%

Class Probabilities:
Low Stress: 3%
Medium Stress: 6%
High Stress: 91%
```

The application also generates personalized wellness suggestions based on user responses.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- SHAP
- Joblib
- Streamlit

---

## 🔮 Future Enhancements

- Real-time stress monitoring
- Mobile application deployment
- Larger and more diverse datasets
- Deep Learning approaches
- Personalized intervention recommendations
- Cloud-based deployment

---

## 👨‍💻 Authors
Hari Krishna, Pragnya Swaminathan

Developed as part of a Machine Learning Project.

### Team Contributions

- Data Preprocessing & Feature Engineering
- Model Development & Evaluation
- Hyperparameter Tuning & Explainable AI
- Application Development & Deployment

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional mental health assessment or counseling.
