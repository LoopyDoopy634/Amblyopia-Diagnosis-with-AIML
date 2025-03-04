import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv("data/amblyopia_dataset_v2.csv")

# Encode categorical variables
label_enc = LabelEncoder()
df['Amblyopia in Family History'] = label_enc.fit_transform(df['Amblyopia in Family History'])
df['Other Eye Disorders in Family History'] = label_enc.fit_transform(df['Other Eye Disorders in Family History'])
df['Amblyopia Diagnosis'] = label_enc.fit_transform(df['Amblyopia Diagnosis'])  # 1 = Positive, 0 = Negative

# Define features and target
X = df[['Patient Age', 'Amblyopia in Family History', 'Other Eye Disorders in Family History']]
y = df['Amblyopia Diagnosis']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
decision_tree.fit(X_train, y_train)

# Save the trained model
with open("models/decision_tree_model.pkl", "wb") as model_file:
    pickle.dump(decision_tree, model_file)

print("Decision Tree model trained and saved as 'models/decision_tree_model.pkl'.")
