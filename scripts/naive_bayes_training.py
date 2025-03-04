import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load the Decision Tree model
with open("models/decision_tree_model.pkl", "rb") as model_file:
    decision_tree = pickle.load(model_file)

# Load the Decision Tree dataset
decision_tree_data = pd.read_csv("data/amblyopia_dataset_v2.csv")
decision_tree_features = decision_tree_data[['Patient Age', 'Amblyopia in Family History', 'Other Eye Disorders in Family History']]

# Encode categorical variables in Decision Tree features
decision_tree_features = decision_tree_features.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

# Get Decision Tree predictions (probabilities)
decision_tree_probabilities = decision_tree.predict_proba(decision_tree_features)

# Load the Bayesian dataset
bayes_data = pd.read_csv("data/bayesian_classifier_data.csv")

# Ensure correct categorical columns are processed
categorical_columns = ['Presence of Strabismus', 'Refractive Error Type', 'Corrective Measures Taken']
for col in categorical_columns:
    bayes_data[col] = LabelEncoder().fit_transform(bayes_data[col])

# Add Decision Tree Positive Probability to the Bayesian dataset
bayes_data['Decision Tree Positive Probability'] = decision_tree_probabilities[:, 1]

# Define features and target for Bayesian training
features = ['Severity of Vision Impairment', 'Presence of Strabismus', 'Refractive Error Type', 
            'Decision Tree Positive Probability', 'Corrective Measures Taken']
target = 'Amblyopia Diagnosis'

X = bayes_data[features]
y = bayes_data[target]

# Train Naive Bayes classifier
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X, y)

# Save the trained model
with open("models/naive_bayes_model.pkl", "wb") as model_file:
    pickle.dump(naive_bayes_model, model_file)

print("Naive Bayes classifier training complete. Model saved.")
