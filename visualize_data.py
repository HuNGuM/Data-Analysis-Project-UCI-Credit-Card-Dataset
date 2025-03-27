import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_catplot(df, kind, x, y=None, hue=None, col=None, output_dir="plots/catplot"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{kind}_{x}"
    if y:
        filename += f"_vs_{y}"
    if hue:
        filename += f"_hue_{hue}"
    if col:
        filename += f"_col_{col}"

    try:
        print(f" Tworzenie wykresu: kind={kind}, x={x}, y={y}, hue={hue}, col={col}")
        g = sns.catplot(
            data=df,
            kind=kind,
            x=x,
            y=y,
            hue=hue,
            col=col,
            height=5,
            aspect=1.3,
            palette="Set2"
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"{kind.upper()} — {y} by {x}" if y else f"{kind.upper()} — count of {x}")
        g.savefig(f"{output_dir}/{filename}.png")
        plt.close()
    except Exception as e:
        print(f" Błąd podczas tworzenia wykresu: {e}")


def plot_error_bars(df, output_dir="plots/errorbars"):
    os.makedirs(output_dir, exist_ok=True)
    target_features = ["LIMIT_BAL", "AGE", "PAY_AMT1"]
    for feature in target_features:
        plt.figure(figsize=(7, 4))
        sns.pointplot(data=df, x="default.payment.next.month", y=feature, hue="SEX", errorbar="sd")
        plt.title(f"Średnia wartość {feature} z błędami (error bars)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/errorbar_{feature}.png")
        plt.close()
    print(" Error bars zapisane do:", output_dir)


def plot_histograms(df, output_dir="plots/histograms"):
    os.makedirs(output_dir, exist_ok=True)
    target_features = ["AGE", "LIMIT_BAL", "PAY_AMT1"]
    for feature in target_features:
        # Histogram podstawowy
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=feature, bins=30, kde=True)
        plt.title(f"Histogram: {feature}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hist_{feature}.png")
        plt.close()

        # Histogram warunkowy (hue)
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=feature, hue="default.payment.next.month", bins=30, kde=True, multiple="stack")
        plt.title(f"Histogram: {feature} z podziałem na default")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hist_{feature}_by_default.png")
        plt.close()
    print(" Histogramy zapisane do:", output_dir)


def plot_correlation_heatmap(df, output_path="plots/heatmap_corr.png"):
    numeric_df = df.select_dtypes(include="number")

    # Obliczenie macierzy korelacji
    corr = numeric_df.corr()

    # Tworzenie heatmapy
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": .8},
        linewidths=0.5
    )
    plt.title(" Korelacja między cechami numerycznymi", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f" Heatmapa korelacji сохранена в: {output_path}")


def plot_linear_regressions(df, output_dir="plots/regression"):
    os.makedirs(output_dir, exist_ok=True)

    pairs = [
        ("AGE", "LIMIT_BAL"),
        ("BILL_AMT1", "PAY_AMT1"),
        ("AGE", "PAY_AMT1"),
        ("LIMIT_BAL", "PAY_AMT1")
    ]

    for x, y in pairs:
        print(f" Linear regression: {x} vs {y}")
        g = sns.lmplot(
            data=df,
            x=x,
            y=y,
            hue="default.payment.next.month",
            height=6,
            aspect=1.3,
            scatter_kws={"s": 20, "alpha": 0.5},
            line_kws={"color": "red"}
        )

        g.fig.suptitle(f"Regresja liniowa: {y} względem {x}", fontsize=14)
        g.fig.subplots_adjust(top=0.92)
        filename = f"regression_{y}_vs_{x}.png"
        g.savefig(f"{output_dir}/{filename}")
        plt.close()

    print(f" Wykresy regresji zapisane do: {output_dir}")
