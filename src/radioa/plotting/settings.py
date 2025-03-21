from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == "__main__":

    models_colors = {
        "SAM": "#8cc5e3",
        "SAM2": "#1a80bb",
        "SAMMED2D": "#bdd373",  #
        "MEDSAM": "#7f7f7f",  #
        "SAMMED3D": "#d8a6a6",  # "#d9f0a3",
        "SAMMED3DTurbo": "#a00000",  # "#78c679",
        "SEGVOL": "#59a89c",  #  "#006837",
    }
    pseudo_bars = {
        "SAM": 0.5,
        "SAM2": 0.7,
        "SAMMED2D": 0.45,
        "SAMMED3D": 0.7,
        "SAMMED3DTurbo": 0.67,
        "MEDSAM": 0.4,
        "SEGVOL": 0.8,
    }

    data = [{"Model": k, "dice": v} for k, v in pseudo_bars.items()]
    df = pd.DataFrame(data)

    # Set color palette for models
    colors = sns.color_palette("viridis", len(pseudo_bars))
    colors = [c for k, c in models_colors.items()]
    hue_order = [k for k, v in models_colors.items()]
    # Use the same color palette as the other plot

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot bars for each model with reduced distance and same colors
    sns.barplot(x="Model", y="dice", data=df, hue="Model", hue_order=hue_order, palette=colors)
    # Add labels and title
    # plt.xlabel('Model')
    plt.title("Box Prompts Per Slice (BPS)", size=30)
    plt.gca().set_facecolor("#f0f0f0")
    plt.ylabel("Average Dice Score", size=25)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.yticks(size=20)
    plt.tight_layout()

    # Display the plot
    plt.savefig("temp.png", dpi=150)
