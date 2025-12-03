# Load the trained model and interactively classify product titles

import joblib
import os
import pandas as pd

# Function to engineer features
def engineer_features(df, text_col="product_title"):
    # Split words once to avoid repeated splitting
    words = df[text_col].str.split()

    df["num_words"] = words.str.len()
    df["num_chars"] = df[text_col].str.len()
    df["has_digits_or_special"] = df[text_col].apply(
        lambda x: int(any(char.isdigit() or not char.isalnum() for char in str(x)))
    )
    df["has_uppercase_terms"] = words.apply(
        lambda ws: int(any(w.isupper() and len(w) > 1 for w in ws))
    )
    df["longest_word_len"] = words.apply(lambda ws: max(len(w) for w in ws))

    return df

# === 1. Load the model ===
model_path = os.path.join(os.path.dirname(__file__), "product_classifier_model.pkl")
pipeline = joblib.load(model_path)

# === 2. Interactive input ===
print("🔍 Enter a product title to classify (or type 'exit' to quit):")

while True:
    title = input("📝 Product title: ")
    if title.lower() == "exit":
        print("👋 Exiting classifier.")
        break
    if not title:
        print("⚠️ Please enter a valid text.")
        continue

    # === 1. Build DataFrame with product_title
    user_input = pd.DataFrame({
        "product_title": [title]
    })

    # === 2. Add engineered features
    user_input = engineer_features(user_input)

    # === 3. Prediction
    predicted_category = pipeline.predict(user_input)[0]

    print(f"📦 Predicted category: {predicted_category}")