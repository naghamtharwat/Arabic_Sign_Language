import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ========= 1. Load CSV files =========
letters_df = pd.read_csv("letters.csv")
numbers_df = pd.read_csv("number.csv")
Words_df = pd.read_csv("Words.csv")

print(f"Letters dataset: {letters_df.shape[0]} samples, {letters_df.shape[1]-1} features")
print(f"Numbers dataset: {numbers_df.shape[0]} samples, {numbers_df.shape[1]-1} features")
print(f"Words dataset: {Words_df.shape[0]} samples, {Words_df.shape[1]-1} features")
# ========= 2. Augmentation Function =========
def augment_sample(row, n=5, noise_level=0.04):
    """Generates new samples by adding random ±noise_level variation."""
    samples = []
    for _ in range(n):
        noise = np.random.normal(0, noise_level * np.random.uniform(0.5, 1.5), len(row) - 1)
        new_row = row[:-1] * (1 + noise)
        samples.append(np.append(new_row, row[-1]))
    return np.array(samples)

def augment_dataset(df, n=5, noise_level=0.04):
    """Apply augmentation to a whole dataset."""
    augmented_data = []
    for _, row in df.iterrows():
        augmented_data.extend(augment_sample(row.values, n=n, noise_level=noise_level))
    aug_df = pd.DataFrame(augmented_data, columns=df.columns)
    df = pd.concat([df, aug_df], ignore_index=True)
    return df

# ========= 3. Train a model (shared logic) =========
def train_model(df, name):
    print(f"\n========== Training {name.upper()} Model ==========")
    
    df = augment_dataset(df, n=6, noise_level=0.04)
    print(f"After augmentation: {len(df)} samples")

    # Split features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=80,   # small enough for mobile, strong enough for accuracy
        max_depth=12,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Model Accuracy: {round(acc * 100, 2)}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    model_path = f"rf_{name.lower()}.pkl"
    scaler_path = f"scaler_{name.lower()}.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {model_path} and {scaler_path}")

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"Top 5 Important Features in {name}:")
    for i in sorted_idx[:5]:
        print(f"  {X.columns[i]}: {importances[i]:.4f}")
    print("====================================")

    return model, scaler, acc

# ========= 4. Train both models =========
letters_model, letters_scaler, acc_letters = train_model(letters_df, "letters")
numbers_model, numbers_scaler, acc_numbers = train_model(numbers_df, "numbers")
Words_model, Words_scaler, acc_Words = train_model(Words_df, "Words")

# ========= 5. Test prediction on example inputs =========
print("\n========== Testing Sample Predictions ==========")
sample_letters = [letters_df.iloc[0, :-1].values]
sample_numbers = [numbers_df.iloc[0, :-1].values]
sample_words = [Words_df.iloc[0, :-1].values]

sample_letters_scaled = letters_scaler.transform(sample_letters)
sample_numbers_scaled = numbers_scaler.transform(sample_numbers)
sample_words_scaled = Words_scaler.transform(sample_words)

pred_letter = letters_model.predict(sample_letters_scaled)[0]
pred_number = numbers_model.predict(sample_numbers_scaled)[0]
pred_word = Words_model.predict(sample_words_scaled)[0]

print(f"Predicted Letter Example: {pred_letter}")
print(f"Predicted Number Example: {pred_number}")
print(f"Predicted Word Example: {pred_word}")

# ========= 6. Summary =========
print("\n========== Training Summary ==========")
print(f"Letters Model Accuracy: {round(acc_letters*100, 2)}%")
print(f"Numbers Model Accuracy: {round(acc_numbers*100, 2)}%")
print(f"Words Model Accuracy: {round(acc_Words*100, 2)}%")
print("Models are ready for integration with your mobile app or ESP32 system.")

