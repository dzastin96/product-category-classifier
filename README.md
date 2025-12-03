# product-category-classifier
Machine learning model for automatic product category classification

## 📌 Overview
This project develops a machine learning pipeline to classify product listings (e.g., CPUs, dishwashers, mobile phones) based on their **title text** and **engineered features**.  
The goal is to deliver an **audit‑ready, interpretable, and production‑ready classifier** that balances semantic depth with structural signals.

---

product-category-classifier/
├── data/
│   ├── final_product_data.pkl
│   └── IMLP4_TASK_03-products.csv
├── model/
│   ├── train_model.py
│   ├── predict_category.py
│   └── product_classifier_model.pkl
├── notebooks/
│   ├── model_training.ipynb
│   └── product_category_analysis.ipynb
└── README.md

---

## ⚙️ Engineered Features
From each `product_title`, we extract:

- `num_words` → total number of words
- `num_chars` → total number of characters
- `has_digits_or_special` → binary flag for digits or special characters
- `has_uppercase_terms` → binary flag for acronyms/uppercase terms (USB, LED, HDMI)
- `longest_word_len` → length of the longest word

---

## 📊 Workflow
1. **Data Cleaning** → normalize product titles, remove duplicates, handle missing values (`product_category_analysis.ipynb`)
2. **Feature Engineering** → add numeric and binary (`product_category_analysis.ipynb`)   
3. **Auditing Feature Relevance** → descriptive stats and boxplots(`product_category_analysis.ipynb`)
4. **Model Training** → Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Support Vector Machine (`model_training.ipynb`)  
5. **Evaluation** → accuracy, macro F1, weighted F1, per‑class precision/recall (`model_training.ipynb`)
6. **Model Selection** → choose best model based on balanced performance and deployment efficiency (`model_training.ipynb`)
7. **Final Model Training** → Train model with best model *Support Vector Machine* (`train_model.py`)
8. **Deployment** → interactive classifier (`predict_category.py`) for real‑time predictions

---

## 🚀 Usage

Run the interactive classifier from `predict_category.py`:

```python
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
