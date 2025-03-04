import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained models
with open("models/decision_tree_model.pkl", "rb") as model_file:
    decision_tree = pickle.load(model_file)

with open("models/naive_bayes_model.pkl", "rb") as model_file:
    naive_bayes_model = pickle.load(model_file)

# Get user input
print("Enter the required details:")
patient_age = int(input("Patient Age: "))
family_history_amblyopia = input("Amblyopia in Family History (Yes/No): ")
family_history_other_disorders = input("Other Eye Disorders in Family History (Yes/No): ")
severity_vision_impairment = float(input("Severity of Vision Impairment (0-1 range): "))
presence_strabismus = input("Presence of Strabismus (Yes/No): ")
refractive_error_type = input("Refractive Error Type (Hyperopia/Myopia/Astigmatism): ")
corrective_measures = input("Corrective Measures Taken (Glasses/Surgery/None): ")

# Prepare Decision Tree input
decision_tree_input = pd.DataFrame({
    'Patient Age': [patient_age],
    'Amblyopia in Family History': [family_history_amblyopia],
    'Other Eye Disorders in Family History': [family_history_other_disorders]
})

# Encode categorical variables
encoder = LabelEncoder()
decision_tree_input['Amblyopia in Family History'] = encoder.fit_transform(decision_tree_input['Amblyopia in Family History'])
decision_tree_input['Other Eye Disorders in Family History'] = encoder.fit_transform(decision_tree_input['Other Eye Disorders in Family History'])

# Get Decision Tree probability
decision_tree_prob = decision_tree.predict_proba(decision_tree_input)[0, 1]

# Prepare Naive Bayes input
naive_bayes_input = pd.DataFrame({
    'Severity of Vision Impairment': [severity_vision_impairment],
    'Presence of Strabismus': [presence_strabismus],
    'Refractive Error Type': [refractive_error_type],
    'Decision Tree Positive Probability': [decision_tree_prob],
    'Corrective Measures Taken': [corrective_measures]
})

# Encode categorical variables
categorical_columns = ['Presence of Strabismus', 'Refractive Error Type', 'Corrective Measures Taken']
for col in categorical_columns:
    naive_bayes_input[col] = encoder.fit_transform(naive_bayes_input[col])

# Get Naive Bayes probability
amblyopia_probability = naive_bayes_model.predict_proba(naive_bayes_input)[0, 1]

# Print results
print(f"Predicted Probability of Amblyopia: {amblyopia_probability:.2f}")
