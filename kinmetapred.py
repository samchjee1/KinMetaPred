import argparse 
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from scipy.special import logit, expit
import warnings
import os

warnings.filterwarnings("ignore")  

feature_columns = ['pph2_score', 'am_score', 'kvp_score']
target_column = 'final_label'

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    smote = SMOTE(random_state=1001)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(test_data[feature_columns])

    return X_train_scaled, y_train_resampled, X_test_scaled, scaler

def train_model(X_train_scaled, y_train_resampled):
    mlp = MLPClassifier(
        random_state=1001,
        early_stopping=True,
        tol=1e-4,
        alpha=0.01
    )

    param_grid = {
        'hidden_layer_sizes': [(300, 150, 75)],
        'activation': ['relu'],
        'solver': ['adam'],
        'learning_rate_init': [0.0001],
    }

    scorer = make_scorer(matthews_corrcoef)

    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=10,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train_scaled, y_train_resampled)

    return grid_search

def calibrate_model(best_model, X_train_scaled, y_train_resampled):
    calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
    calibrated.fit(X_train_scaled, y_train_resampled)
    return calibrated

def scale_probabilities(probs, factor=2.5):
    logits = logit(probs)
    scaled_logits = logits * factor
    return expit(scaled_logits)

def evaluate(test_data, probs_original, probs_scaled, pred_scaled, output_prefix):
    y_test = test_data[target_column]
    mcc = matthews_corrcoef(y_test, pred_scaled)
    return mcc  # Only return MCC

def save_results(test_data, predictions, confidences, grid_search, output_prefix):
    test_data['Predicted_Label'] = predictions
    test_data['Confidence_Score'] = confidences

    output_file = f"{output_prefix}_predictions.csv"
    test_data.to_csv(output_file, index=False)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(f"{output_prefix}_cv_scores.csv", index=False)

    if target_column in test_data.columns:
        mcc = evaluate(test_data, grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1],
                       confidences, predictions, output_prefix)
        with open(f"{output_prefix}_results.txt", "w") as f:
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Test MCC: {mcc:.4f}\n")  # Only write MCC
        print(f"Results available in {output_file} and {output_prefix}_results.txt")
    else:
        print(f"Predictions available in {output_file} (no labels in test data).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict with MLP + calibration.")
    parser.add_argument("--train", required=True, help="Training CSV file")
    parser.add_argument("--test", required=True, help="Test CSV file")
    parser.add_argument("--output", default="mlp_output", help="Prefix for output files")

    args = parser.parse_args()

    # Load and process data
    train_df, test_df = load_data(args.train, args.test)
    print("Class distribution in training set:")
    print(train_df[target_column].value_counts(normalize=True))

    X_train_scaled, y_train_resampled, X_test_scaled, scaler = preprocess_data(train_df, test_df)

    # Train and calibrate model
    grid_search = train_model(X_train_scaled, y_train_resampled)
    best_model = grid_search.best_estimator_
    calibrated_model = calibrate_model(best_model, X_train_scaled, y_train_resampled)

    # Predict and scale probabilities
    probs = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    scaled_probs = scale_probabilities(probs)
    predictions = (scaled_probs >= 0.5).astype(int)

    # Save results
    save_results(test_df, predictions, scaled_probs, grid_search, args.output)
