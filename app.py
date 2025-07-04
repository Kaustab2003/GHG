import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input
from fpdf import FPDF
import smtplib
from email.message import EmailMessage
import os

# Load model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Supply Chain Emissions Prediction")

st.markdown("""
This app predicts **Supply Chain Emission Factors with Margins** based on DQ metrics and other parameters.
""")

# Input form
with st.form("prediction_form"):
    substance = st.selectbox("Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
    unit = st.selectbox("Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
    source = st.selectbox("Source", ['Commodity', 'Industry'])
    supply_wo_margin = st.number_input("Supply Chain Emission Factors without Margins", min_value=0.0)
    margin = st.number_input("Margins of Supply Chain Emission Factors", min_value=0.0)
    dq_reliability = st.slider("DQ Reliability", 0.0, 1.0)
    dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0)
    dq_geo = st.slider("DQ Geographical Correlation", 0.0, 1.0)
    dq_tech = st.slider("DQ Technological Correlation", 0.0, 1.0)
    dq_data = st.slider("DQ Data Collection", 0.0, 1.0)

    email_address = st.text_input("Enter your email to receive the report:")

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'Substance': substance,
        'Unit': unit,
        'Supply Chain Emission Factors without Margins': supply_wo_margin,
        'Margins of Supply Chain Emission Factors': margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
        'DQ DataCollection of Factors without Margins': dq_data,
        'Source': source,
    }

    input_df = preprocess_input(pd.DataFrame([input_data]))
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")

    # Emission graphs using Streamlit bar chart
    st.subheader("Emission Breakdown (Bar Chart)")
    data = pd.DataFrame({
        'Category': ['Without Margin', 'Margin', 'Predicted Total'],
        'Value': [supply_wo_margin, margin, prediction[0]]
    })
    st.bar_chart(data.set_index('Category'))

    st.subheader("Emission Distribution (Pie Chart)")
    pie_data = pd.DataFrame({
        'Category': ['Without Margin', 'Margin'],
        'Value': [supply_wo_margin, margin]
    })
    st.write(pie_data)

    # Save results as CSV
    result_df = pd.DataFrame({
        'Substance': [substance],
        'Unit': [unit],
        'Source': [source],
        'Supply Chain Emission Factors without Margins': [supply_wo_margin],
        'Margins of Supply Chain Emission Factors': [margin],
        'Predicted Supply Chain Emission Factor with Margin': [prediction[0]]
    })

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Result as CSV", data=csv, file_name="emission_prediction.csv", mime='text/csv')

    # Create PDF
    def create_pdf(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Emission Prediction Report", ln=True, align='C')
        for key, value in data.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
        path = "result.pdf"
        pdf.output(path)
        return path

    pdf_path = create_pdf({**input_data, "Predicted Total Emission": prediction[0]})

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(label="Download Report as PDF", data=pdf_file, file_name="emission_report.pdf", mime="application/pdf")

    # Email Function
    def send_email(receiver_email, subject, body, attachment_path):
        sender_email = "your_email@gmail.com"  # Replace with your email
        sender_password = "your_app_password"  # Use an app password or OAuth

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.set_content(body)

        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(f.name)

        msg.add_attachment(file_data, maintype='application', subtype='pdf', filename=file_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

    if email_address:
        try:
            send_email(email_address, "Your Emission Prediction Report", "Please find your report attached.", pdf_path)
            st.success(f"Email sent to {email_address}")
        except Exception as e:
            st.error(f"Error sending email: {e}")