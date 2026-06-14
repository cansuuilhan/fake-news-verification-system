import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_single_model(model_name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(
            y_test,
            y_pred,
            target_names=["Sahte", "Gerçek"],
            zero_division=0
        )
    }


def run_evaluation():
    print("=" * 70)
    print("FAKESENSE AKADEMİK MODEL KARŞILAŞTIRMA TESTİ")
    print("=" * 70)

    dataset_path = "data/fake_real_tr/turkish_fake_real.csv"
    text_col = "clean_data"
    label_col = "label"

    df = pd.read_csv(dataset_path)
    df = df[[text_col, label_col]].dropna()

    X = df[text_col].astype(str).str.lower()
    y = df[label_col]

    print(f"Toplam veri sayısı: {len(df)}")
    print("\nEtiket dağılımı:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC()
    }

    results = []

    print("\n" + "=" * 70)
    print("80/20 TRAIN-TEST MODEL SONUÇLARI")
    print("=" * 70)

    for model_name, model in models.items():
        result = evaluate_single_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

        print(f"\nModel: {model_name}")
        print("-" * 70)
        print(f"Accuracy  : %{result['Accuracy'] * 100:.2f}")
        print(f"Precision : %{result['Precision'] * 100:.2f}")
        print(f"Recall    : %{result['Recall'] * 100:.2f}")
        print(f"F1-Score  : %{result['F1-Score'] * 100:.2f}")
        print("\nConfusion Matrix:")
        print(result["Confusion Matrix"])
        print("\nClassification Report:")
        print(result["Classification Report"])

    summary_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": round(r["Accuracy"] * 100, 2),
            "Precision": round(r["Precision"] * 100, 2),
            "Recall": round(r["Recall"] * 100, 2),
            "F1-Score": round(r["F1-Score"] * 100, 2)
        }
        for r in results
    ])

    print("\n" + "=" * 70)
    print("MODEL KARŞILAŞTIRMA ÖZETİ")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("5-FOLD CROSS VALIDATION SONUÇLARI")
    print("=" * 70)

    cv_results = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2)
            )),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        cv_results.append({
            "Model": model_name,
            "Fold-1": round(scores[0] * 100, 2),
            "Fold-2": round(scores[1] * 100, 2),
            "Fold-3": round(scores[2] * 100, 2),
            "Fold-4": round(scores[3] * 100, 2),
            "Fold-5": round(scores[4] * 100, 2),
            "Ortalama Accuracy": round(scores.mean() * 100, 2)
        })

    cv_df = pd.DataFrame(cv_results)
    print(cv_df.to_string(index=False))

    summary_df.to_csv(
        "model_comparison_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    cv_df.to_csv(
        "cross_validation_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSonuçlar kaydedildi:")
    print("- model_comparison_results.csv")
    print("- cross_validation_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()