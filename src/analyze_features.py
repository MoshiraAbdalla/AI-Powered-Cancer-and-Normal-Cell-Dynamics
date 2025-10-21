import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_features(features_csv, output_dir):
    """
    Perform exploratory data analysis on extracted cell motion features.
    Generates:
      - Histograms for each feature
      - Boxplots comparing distribution spread
      - Scatter plot for speed vs. persistence
      - Summary statistics table (mean ± std)
    """

    # --- Step 1: Load feature data ---
    df = pd.read_csv(features_csv)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loaded {len(df)} cell feature entries from {features_csv}")

    # --- Step 2: Basic summary statistics ---
    summary = df.describe()[['mean_speed(px/s)', 'total_displacement(px)',
                             'mean_squared_displacement(px²)',
                             'directional_persistence', 'mean_turn_angle(deg)']].T
    summary['mean ± std'] = summary['mean'].round(2).astype(str) + " ± " + summary['std'].round(2).astype(str)
    print("\n[SUMMARY STATISTICS]\n")
    print(summary[['mean ± std']])

    summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

    # --- Justification ---
    # Basic descriptive stats give a quick sense of central tendency and variability.
    # For biological motion analysis, mean ± std is preferred since trajectories vary naturally.

    # --- Step 3: Plot histograms of major motion features ---
    features_to_plot = ['mean_speed(px/s)', 'directional_persistence', 'mean_turn_angle(deg)']
    plt.style.use('seaborn-v0_8-whitegrid')

    for feature in features_to_plot:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], kde=True, bins=20, color='royalblue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{feature.replace('/', '_')}_hist.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    # --- Justification ---
    # Histograms show population-level variability among cells.
    # KDE smoothing (kernel density estimate) highlights multimodal behavior (e.g., subpopulations of motile cells).

    # --- Step 4: Boxplots for variability comparison ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[['mean_speed(px/s)', 'directional_persistence', 'mean_turn_angle(deg)']],
                palette="Set2")
    plt.title("Feature Variability in Normal Cells")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_features.png"), dpi=300)
    plt.close()

    # --- Step 5: Scatter plot (Speed vs. Persistence) ---
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x='mean_speed(px/s)', y='directional_persistence', data=df, s=40, color='darkred')
    plt.title("Speed vs. Directional Persistence (Normal Cells)")
    plt.xlabel("Mean Speed (px/s)")
    plt.ylabel("Directional Persistence")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_speed_persistence.png"), dpi=300)
    plt.close()

    # --- Justification ---
    # Scatter plots reveal relationship patterns between motility rate and trajectory straightness.
    # For normal cells, you often expect a moderate positive trend:
    # faster cells tend to move more persistently.

    # --- Step 6: Save cleaned dataset for later group comparison ---
    df.to_csv(os.path.join(output_dir, "normal_features_clean.csv"), index=False)
    print(f"\n[INFO] Analysis complete. Figures and summary saved in '{output_dir}'.")


if __name__ == "__main__":
    features_csv = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\features\normal_features.csv"
    output_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\analysis\normal"

    analyze_features(features_csv, output_dir)
