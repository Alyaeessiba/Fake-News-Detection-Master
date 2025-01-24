import streamlit as st
import joblib

# Load the vectorizer and models
vectorizer = joblib.load("vectorizer.jb")
lr_model = joblib.load("lr_model.jb")
dtc_model = joblib.load("DTC_model.jb")
rfc_model = joblib.load("rfc_model.jb")
gbc_model = joblib.load("gbc_model.jb")

# Streamlit app title and description
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

# User input for news article
inputn = st.text_area("News Article:", "")

# Dropdown to select a model
model_choice = st.selectbox(
    "Choose a model to use:",
    ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
)

# Model selection logic
if model_choice == "Logistic Regression":
    model = lr_model
elif model_choice == "Decision Tree":
    model = dtc_model
elif model_choice == "Random Forest":
    model = rfc_model
else:  # Gradient Boosting
    model = gbc_model

# Check news button
if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")
