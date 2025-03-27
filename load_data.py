import pandas as pd

def load_dataset(path="UCI_Credit_Card.csv"):

    try:
        df = pd.read_csv(path)
        print(" Zbiór danych został pomyślnie załadowany")
        return df
    except FileNotFoundError:
        print(f" Plik nie został znaleziony pod ścieżką: {path}")
        return None
    except Exception as e:
        print(f" Błąd podczas ładowania: {e}")
        return None

if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        print(" Pierwsze 5 wierszy zbioru danych:")
        print(df.head())
