# ---------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------
# LOAD DATASET
# ---------------------------------------------
df = pd.read_csv('Dataset/eggplant_varieties.csv')

# Features and Labels
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50
)

# ---------------------------------------------
# FEATURE SCALING
# ---------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------
# MODEL TRAINING
# ---------------------------------------------
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# ---------------------------------------------
# SAVE MODEL + SCALER
# ---------------------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))


# ---------------------------------------------
# FUNCTION 1: Predict crop for a SINGLE INPUT
# ---------------------------------------------
def predict_single(input_list):
    """
    input_list = [N, P, K, temperature, humidity, ph, rainfall]
    """
    input_scaled = scaler.transform([input_list])
    prediction = model.predict(input_scaled)
    return prediction[0]


# ---------------------------------------------
# FUNCTION 2: Predict crop for ALL TEST DATA
# ---------------------------------------------
def predict_all_test_data():
    predictions = model.predict(X_test_scaled)
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })
    return results


# ---------------------------------------------
# FUNCTION 3: Predict ALL POSSIBLE CROPS
# ---------------------------------------------
def get_all_possible_predictions():
    return sorted(list(df["label"].unique()))


# ---------------------------------------------
# FUNCTION 4: TOP-K PREDICTIONS (LABELS ONLY)
# ---------------------------------------------
def predict_top_k(input_list, k=3):
    """
    Returns the top K most likely crop labels.
    No probabilities returned.
    """
    input_scaled = scaler.transform([input_list])

    # Predict full probability distribution
    probabilities = model.predict_proba(input_scaled)[0]

    # Get top K indices (sorted)
    top_k_indices = probabilities.argsort()[-k:][::-1]

    # Return only labels
    top_k_labels = [model.classes_[i] for i in top_k_indices]

    return top_k_labels


# --------------------------------------------------
# EXAMPLE USAGE (COMMENTED)
# --------------------------------------------------
# new_features = [117, 32, 34, 26.27, 52.12, 6.75, 127.17]
# print("Top 3 Crops:", predict_top_k(new_features, k=3))
