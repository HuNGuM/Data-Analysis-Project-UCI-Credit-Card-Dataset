import pandas as pd
from load_data import load_dataset

def analyze_data(df):
    """
    Przeprowadza opisową analizę danych:
    - Dla cech numerycznych: średnia, mediana, min, max, odchylenie standardowe, 5 i 95 percentyl, brakujące wartości.
    - Dla cech kategorialnych: liczba unikalnych klas, brakujące dane, udział najczęstszej klasy.
    """

    # Cechy kategorialne według znaczenia, nawet jeśli są liczbowe
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE", "default.payment.next.month"]
    numerical_cols = [col for col in df.columns if col not in categorical_cols and col != "ID"]

    print("\n Analiza cech numerycznych...")
    numeric = df[numerical_cols]
    num_stats = numeric.describe(percentiles=[0.05, 0.95]).T
    num_stats["median"] = numeric.median()
    num_stats["missing_values"] = numeric.isnull().sum()

    num_stats = num_stats[
        ["count", "mean", "median", "min", "5%", "50%", "95%", "max", "std", "missing_values"]
    ].rename(columns={"5%": "percentile_5", "95%": "percentile_95", "std": "std_dev"})

    num_stats.to_csv("numeric_summary.csv")
    print(" Zapisano do pliku numeric_summary.csv")

    print("\n Analiza cech kategorialnych...")
    categorical = df[categorical_cols]

    cat_stats = pd.DataFrame({
        "unique_classes": categorical.nunique(),
        "missing_values": categorical.isnull().sum(),
        "top_class_ratio": categorical.apply(lambda col: col.value_counts(normalize=True).max())
    })

    cat_stats.to_csv("categorical_summary.csv")
    print(" Zapisano do pliku categorical_summary.csv")

if __name__ == "__main__":
    df = load_dataset("UCI_Credit_Card.csv")
    if df is not None:
        analyze_data(df)
