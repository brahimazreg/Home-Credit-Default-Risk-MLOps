import requests
import streamlit as st

st.title("Home Credit Default Risk Predictor")

sample = requests.get("http://localhost:8000/sample").json()["data"]

with st.form("predict_form"):

    amt_income = st.number_input("Income", min_value=0, value=200000)
    amt_credit = st.number_input("Credit Amount", min_value=0, value=400000)
    cnt_children = st.number_input("Children", min_value=0, value=0)

    gender = st.selectbox("Gender", ["M", "F"])
    owns_car = st.selectbox("Owns Car", ["Y", "N"])
    owns_realty = st.selectbox("Owns Realty", ["Y", "N"])

    submitted = st.form_submit_button("Predict")

if submitted:

    payload = {
        "data": {
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT": amt_credit,
            "CNT_CHILDREN": cnt_children,
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": owns_car,
            "FLAG_OWN_REALTY": owns_realty
        }
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=payload
    )

    result = response.json()

    st.write("RAW:", result)

    if response.status_code != 200:
        st.error(result.get("detail", "API error"))
    elif "prediction" not in result:
        st.error("Missing prediction in response")
    else:
        st.success("Prediction received")
        st.write("Prediction:", result["prediction"])
        st.write("Probability:", result["probability"])
    

      