from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, send_from_directory
import os


app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('Dataset Collection - Form Responses 1.csv')

# Encode categorical data
le = LabelEncoder()
df['Age'] = le.fit_transform(df['Age'])
df['Blood Type'] = le.fit_transform(df['Blood Type'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Type of Wound'] = le.fit_transform(df['Type of Wound'])
df['Wound Size'] = le.fit_transform(df['Wound Size'])
df['Medication Taken'] = le.fit_transform(df['Medication Taken'])

# Extract features and target variable
X = df.drop(columns=['Time Taken to Heal(in days)'])
y = df['Time Taken to Heal(in days)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Split data and train the RandomForest model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model for future use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

app = Flask(__name__)

# Homepage with the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    age = request.form['age']
    blood_type = request.form['blood_type']
    gender = request.form['gender']
    wound_type = request.form['wound_type']
    wound_size = request.form['wound_size']
    medication = request.form['medication']
    
    # Simple logic for prediction based on wound size and medication
    if wound_size == 'Small':
        prediction = '1-2 days' if medication == 'Yes' else '2-4 days'
    elif wound_size == 'Medium':
        prediction = '3-5 days' if medication == 'Yes' else '4-6 days'
    elif wound_size == 'Large':
        prediction = '5-7 days' if medication == 'Yes' else 'More than a week'
    else:
        prediction = 'Unknown'

    # Pass the prediction to the result template
    return render_template('result.html', prediction=prediction)

# Route for Data Analytics (placeholder)
@app.route('/analytics')
def analytics():
    # Render your data analytics page 
    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True)
