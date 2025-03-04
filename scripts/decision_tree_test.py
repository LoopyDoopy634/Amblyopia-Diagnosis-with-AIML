import pickle
import numpy as np
import pandas as pd

# Load the trained Decision Tree model
with open("models/decision_tree_model.pkl", "rb") as model_file:
    decision_tree = pickle.load(model_file)

# Function to predict using the Decision Tree
def predict_amblyopia(patient_age, amblyopia_family_history, other_eye_disorders_family_history):
    input_data = pd.DataFrame([[patient_age, amblyopia_family_history, other_eye_disorders_family_history]],
                              columns=['Patient Age', 'Amblyopia in Family History', 'Other Eye Disorders in Family History'])
    probabilities = decision_tree.predict_proba(input_data)
    
    print(f"Prediction Probabilities: Negative: {probabilities[0][0]*100:.2f}%, Positive: {probabilities[0][1]*100:.2f}%")

# Take user input
if __name__ == "__main__":
    patient_age = int(input("Enter patient age: "))
    amblyopia_family_history = int(input("Amblyopia in family history? (Yes=1, No=0): "))
    other_eye_disorders_family_history = int(input("Other eye disorders in family history? (Yes=1, No=0): "))
    
    predict_amblyopia(patient_age, amblyopia_family_history, other_eye_disorders_family_history)
