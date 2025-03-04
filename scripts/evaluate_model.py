import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns
import os

def evaluate_model(model_path, test_data_folder, results_folder):
    # Load the model
    model = load_model(model_path)

    # Load test data
    X_test = np.load(os.path.join(test_data_folder, "X_train.npy"))
    y_test = np.load(os.path.join(test_data_folder, "y_train.npy"))

    # Evaluate loss and accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Amblyopia", "Amblyopia"], yticklabels=["No Amblyopia", "Amblyopia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_folder, "confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(y_test, y_pred_classes, target_names=["No Amblyopia", "Amblyopia"])
    print(report)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(results_folder, "roc_curve.png"))
    plt.close()

    # Save results to a text file
    with open(os.path.join(results_folder, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"AUC: {auc}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    print(f"Results saved to {results_folder}")

if __name__ == "__main__":
    model_path = "models/amblyopia_detection_model.h5"
    test_data_folder = "data/processed"
    results_folder = "results"

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    evaluate_model(model_path, test_data_folder, results_folder)