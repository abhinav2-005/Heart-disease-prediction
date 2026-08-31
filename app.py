import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")
# print(expected_columns)

st.title("Heart Stroke Prediction")
st.markdown("Provide the following details to check your heart stroke risk:")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):
    input_df = pd.DataFrame(columns=expected_columns)
    input_df.loc[0] = 0

    # Numerical features
    input_df.at[0, 'Age'] = age
    input_df.at[0, f"Sex"] = 1
    input_df.at[0, 'RestingBP'] = resting_bp
    input_df.at[0, 'Cholesterol'] = cholesterol
    input_df.at[0, 'FastingBS'] = fasting_bs
    input_df.at[0, 'MaxHR'] = max_hr
    input_df.at[0, 'Oldpeak'] = oldpeak

    # Categorical (one-hot)
    input_df.at[0, f"ChestPainType_{chest_pain}"] = 1
    input_df.at[0, f"RestingECG_{resting_ecg}"] = 1
    input_df.at[0, f"ExerciseAngina_{exercise_angina}"] = 1
    input_df.at[0, f"ST_Slope_{st_slope}"] = 1


    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    # Scale
    
    cols = ["Age","RestingBP","Cholesterol","MaxHR"]

    input_df[cols] = scaler.transform(input_df[cols])

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")