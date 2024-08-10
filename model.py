import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

data=pd.read_csv(r'D:\assignments dsa\HTML_vs-code\loan_prediction\train_ctrUa4K.csv')

data = data.drop(columns=['Loan_ID'])

for i in ['Gender','Married','Dependents','Self_Employed']:
    data[i].fillna(data[i].mode()[0],inplace=True)

for i in ['LoanAmount','Loan_Amount_Term','Credit_History']:
  data[i].fillna(data[i].median(),inplace=True)

label_encoders = {}
for column in ['Gender', 'Married', 'Education','Dependents', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])


X = data.drop(columns=['Loan_Status'],axis=1)
y = data['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled,y)

with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and preprocessing tools saved successfully!")
