from load_data import load_dataset
from analyze_data import analyze_data
from visualize_data import (
    save_catplot,
    plot_error_bars,
    plot_histograms,
    plot_correlation_heatmap,
    plot_linear_regressions
)

def main():
    df = load_dataset("UCI_Credit_Card.csv")
    if df is None:
        return

    # === 1. Analiza danych ===
    analyze_data(df)

    # === 2. Wykresy kategorialne ===
    category = "default.payment.next.month"
    hue_col = "SEX"
    col_split = "EDUCATION"
    features = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_0", "PAY_AMT1"]

    for feature in features:
        for kind in ["box", "violin"]:
            save_catplot(df, kind=kind, x=category, y=feature, hue=hue_col)
        save_catplot(df, kind="violin", x=category, y=feature, hue=hue_col, col=col_split)

    save_catplot(df, kind="count", x="EDUCATION", hue="SEX")
    save_catplot(df, kind="count", x="MARRIAGE", hue="default.payment.next.month")

    # === 3. Wykresy error bars i histogramy ===
    plot_error_bars(df)
    plot_histograms(df)

    # === 4. Heatmapa i regresja liniowa ===
    plot_correlation_heatmap(df)
    plot_linear_regressions(df)

    print(" Cała analiza została pomyślnie zakończona")

if __name__ == "__main__":
    main()
