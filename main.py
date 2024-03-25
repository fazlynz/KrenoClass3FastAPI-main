import pickle
import pandas as pd
import time  # Import the time module
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

class LoanApplication(BaseModel):
    Loan_ID: Union[str, None]
    Gender: Union[str, None]
    Married: Union[str, None]
    Dependents: Union[str, None]
    Education: Union[str, None]
    Self_Employed: Union[str, None]
    ApplicantIncome: Union[int, None]
    CoapplicantIncome: Union[float, None]
    LoanAmount: Union[float, None]
    Loan_Amount_Term: Union[float, None]
    Credit_History: Union[float, None]
    Property_Area: Union[str, None]

app = FastAPI()

@app.post("/predict")
async def predict(request: LoanApplication):
    start_time = time.time()  # Record the start time

    #data = request.dict()  # Use .dict() instead of .model_dump() to get the input data as a dictionary
    data = request.model_dump()
    df = pd.DataFrame([data])

    # Fill empty values for Credit_History column with value 1
    df['Credit_History'].fillna(1, inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['Dependents'].fillna("0", inplace=True)
    df['Gender'].fillna("Male", inplace=True)

    # Remove any data that still has null value
    df = df.dropna()

    if df.empty:
        return {"error": "Data is not in a valid format"}

    # Data processing and mapping
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Married'] = df['Married'].apply(lambda x: 1 if x == 'Married' else 0)
    df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})  # Domain knowledge feature
    df['Education'] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df['Self_Employed'] = df['Self_Employed'].apply(lambda x: 1 if x == 'No' else 0)
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Rural': 0, 'Semiurban': 1})  # Ordinal feature

    # Load model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make prediction
    loan_id = df.iloc[0]['Loan_ID']
    input_data = df.drop(columns=['Loan_ID'])
    prediction = model.predict(input_data)
    prediction = prediction.tolist()

    # Additional output
    status = 1 if data.get("status") == "success" else 0
    error_code = 0 if data.get("status") == "success" else 1

    # Get prediction label and value
    prediction_label = 1  # Set prediction label to 1
    prediction_value = "approve"  # Set prediction value to "approve"

    # Calculate time taken for prediction
    end_time = time.time()
    time_taken = f"{int((end_time - start_time) * 1000)}ms"  # Convert time to milliseconds

    # Prepare output
    return {
        "loan_id": loan_id,
        "predicted_class": prediction[0],
        "predicted_class_name": "Approve" if prediction[0] == 1 else "Reject",
        "status": status,
        "error_code": error_code,
        "prediction": {
            "label": prediction_label,
            "value": prediction_value
        },
        "time_taken": time_taken  # Set time_taken based on the duration of the prediction process
    }
