import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from utils.preprocessor import preprocess_input

# Example: Create sample data --------------------------------------------
data = pd.DataFrame({
    'Substance': ['carbon dioxide', 'methane', 'nitrous oxide', 'carbon dioxide', 'other GHGs'],
    'Unit': ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price', 'kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'],
    'Supply Chain Emission Factors without Margins': [12, 30, 15, 8, 22],
    'Margins of Supply Chain Emission Factors': [2, 5, 3, 1, 4],
    'DQ ReliabilityScore of Factors without Margins': [0.8, 0.7, 0.9, 0.85, 0.75],
    'DQ TemporalCorrelation of Factors without Margins': [0.6, 0.5, 0.8, 0.7, 0.55],
    'DQ GeographicalCorrelation of Factors without Margins': [0.7, 0.6, 0.9, 0.65, 0.7],
    'DQ TechnologicalCorrelation of Factors without Margins': [0.9, 0.8, 0.85, 0.95, 0.88],
    'DQ DataCollection of Factors without Margins': [0.75, 0.65, 0.8, 0.78, 0.7],
    'Source': ['Commodity', 'Industry', 'Commodity', 'Industry', 'Commodity'],
})
# Create a DataFrame of results
result_df = pd.DataFrame({
    'Substance': [substance],
    'Unit': [unit],
    'Source': [source],
    'Supply Chain Emission Factors without Margins': [supply_wo_margin],
    'Margins of Supply Chain Emission Factors': [margin],
    'Predicted Supply Chain Emission Factor with Margin': [prediction[0]]
})

# Download as CSV
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Result as CSV", data=csv, file_name="emission_prediction.csv", mime='text/csv')

# For PDF, you need to install fpdf first:
# pip install fpdf

from fpdf import FPDF

def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Emission Prediction Report", ln=True, align='C')
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    # Save the PDF to a file
    pdf.output("result.pdf")
    return "result.pdf"

if st.button("Download Report as PDF"):
    pdf_path = create_pdf(data)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(label="Download PDF", data=pdf_file, file_name="emission_report.pdf", mime="application/pdf")


import smtplib
from email.message import EmailMessage

def send_email(receiver_email, subject, body, attachment_path):
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(body)

    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        file_name = f.name

    msg.add_attachment(file_data, maintype='application', subtype='pdf', filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

if st.button("Email Report"):
    email_address = st.text_input("Enter your email to receive the report:")
    pdf_file = create_pdf(data)
    if email_address:
        send_email(email_address, "Your Emission Prediction Report", "Please find your report attached.", pdf_file)
        st.success(f"Email sent to {email_address}")



# Target Variable (Example: some made-up emission factors WITH margin)
target = [14, 37, 18, 10, 26]

# Preprocess categorical columns -----------------------------------------
data_preprocessed = preprocess_input(data)

# Scale data ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_preprocessed)

# Train model ----------------------------------------------------------
model = LinearRegression()
model.fit(X_scaled, target)

# Save model and scaler ------------------------------------------------
# Ensure 'models' folder exists
import os
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/LR_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Model and scaler saved in 'models/' folder.")
