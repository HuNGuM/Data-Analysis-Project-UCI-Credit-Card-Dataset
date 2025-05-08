import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from my_linear_regression_closed import linear_regression_closed_form, predict as predict_closed
from my_linear_regression_gd import gradient_descent, predict as predict_gd
from using_sklearn import sklearn_fit

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

    df.columns = df.columns.str.strip()
    X = df[["PAY_AMT1"]].to_numpy()
    y = df["BILL_AMT1"].to_numpy()

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    theta_closed = linear_regression_closed_form(X_train, y_train)
    y_pred_closed = predict_closed(X_test, theta_closed)
    print("Closed-form MSE:", mean_squared_error(y_test, y_pred_closed))

    theta_gd = gradient_descent(X_train, y_train, lr=1e-7, epochs=1000)
    y_pred_gd = predict_gd(X_test, theta_gd)
    print("Gradient Descent MSE:", mean_squared_error(y_test, y_pred_gd))

    y_pred_sk = sklearn_fit(X_train, y_train, X_test)
    print("scikit-learn MSE:", mean_squared_error(y_test, y_pred_sk))


if __name__ == "__main__":
    main()
