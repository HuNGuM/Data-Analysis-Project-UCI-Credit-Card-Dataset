import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from model_pipeline import get_models, build_pipeline

def load_dataset(path="UCI_Credit_Card.csv"):
    try:
        df = pd.read_csv(path)
        print(df.columns.tolist())
        print("Dane zostały załadowane.")
        return df
    except Exception as e:
        print(f"Błąd przy ładowaniu danych: {e}")
        return None

def main():
    df = load_dataset()
    if df is None:
        return

    target = "default.payment.next.month"
    X = df.drop(columns=["ID", target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()

    for name, model in models.items():
        print(f"\nTrenowanie modelu: {name}")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
