# Train and save the best-performing model (SVM with product_title only)

import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import os
import pandas as pd

# Load data
try:
    # Get absolute path to the data folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "final_product_data.pkl")
    save_path = os.path.join(os.path.dirname(__file__), "product_classifier_model.pkl")
    print(f"Loading data from: {data_path}")
    df = joblib.load(data_path)
    print("✅ Loaded with joblib")
except:
    print("❌ Failed to load data")

print(df.head(5))

# Prepare features and labels
preprocessor = ColumnTransformer([
    ('title', TfidfVectorizer(), 'product_title'),
    ('length', MinMaxScaler(), [
        "longest_word_len",
        "num_words",
        "num_chars"
    ]),
    ('binary', MinMaxScaler(), [
        "has_digits_or_special",
        "has_uppercase_terms"
    ]),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearSVC(max_iter=2000))
])

X = df[['product_title', 'num_words', 'num_chars', 'has_digits_or_special', 'has_uppercase_terms', 'longest_word_len']]
y = df['category_label'].astype(str)

pipeline.fit(X, y)


# Save model
joblib.dump(pipeline, save_path)
print("✅ Model trained and saved as 'product_classifier_model.pkl'")
