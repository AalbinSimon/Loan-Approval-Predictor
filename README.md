# Loan-Approval-Predictor
Developed an app using multiple classification model to predict the loan approval. To check the result multiple matrix has been checked 
This project focuses on predicting whether a loan applicant will default using supervised machine learning techniques.
It includes EDA, preprocessing, feature engineering, pipeline automation, model comparison, ROC–AUC evaluation, and a deployed Streamlit app.

📌 Project Overview

Financial institutions face major risks due to loan defaults.
By predicting default probability early, banks can make better lending decisions and reduce losses.

This project builds a complete machine learning workflow to classify loan applicants as Approved (1) or Rejected (0) based on demographic, financial, and credit-related features.

⭐ Key Features

✔ Complete EDA with visual insights
✔ Data cleaning & imputation using ColumnTransformer
✔ OneHotEncoding + Scaling inside a unified Pipeline
✔ Multiple model training & AUC comparison
✔ Selected the best-performing model (Logistic Regression)
✔ ROC–AUC curve plotted
✔ Exported model + preprocessor using joblib
✔ Built a fully interactive Streamlit Web App for predictions

📂 Project Structure
📁 Loan-Default-Prediction
│
├── 📄 training.py          # Model training script
├── 📄 app.py               # Streamlit app
├── 📄 loan_model.pkl       # Saved best ML model
├── 📄 preprocessor.pkl     # Saved preprocessing pipeline
├── 📄 requirements.txt     # Dependencies
├── 📄 README.md            # Project documentation
└── 📁 data
      └── train.csv        # Original dataset

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Joblib

📊 Machine Learning Models Used
Model	ROC–AUC Score
Logistic Regression	0.86
Decision Tree	0.74
Random Forest	0.82
KNN	0.85

➡ Logistic Regression was chosen for deployment due to its stable performance and interpretability.

🔁 Preprocessing Workflow

The following transformations were automated using ColumnTransformer + Pipeline:

Missing value imputation

OneHotEncoding of categorical variables

Scaling numerical variables

Passing untouched columns using remainder='passthrough'

This ensured clean, reproducible, production-ready preprocessing.

🚀 Streamlit App

The interactive app allows users to:

Enter applicant details

Process data via the saved preprocessor

Predict loan approval with the trained ML model

Run using:

streamlit run app.py

📚 Learnings & Reflection

This project strengthened my understanding of how essential EDA, preprocessing, pipelines, and ColumnTransformers are for building robust ML systems.
Since the dataset was relatively small, performance was strong even without hyperparameter tuning. I plan to extend this by experimenting with a larger dataset where I can include hyperparameter tuning and further evaluate model improvements.

📎 Results

✔ Achieved high model performance without tuning
✔ Built clean, modular, production-quality code
✔ Successfully deployed a real-time prediction app


<img width="1916" height="1016" alt="image" src="https://github.com/user-attachments/assets/84b601aa-ad8a-4961-978d-950cf475fffa" />




