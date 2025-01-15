import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from radioa.eval.crawl_all_res import prompter_names, models_colors

# Define grey shades for LaTeX table background
GREY_SHADES: Dict[str, str] = {
    'SamMed 3D': '\\rowcolor[gray]{0.95}',
    'SamMed 3D Turbo': '\\rowcolor[gray]{0.9}',
    ' SegVol': '\\rowcolor[gray]{1}',
}

# ========== Utility Functions ==========

def generate_latex(df: pd.DataFrame, grey_shades: Dict[str, str]) -> str:
    """
    Generates a LaTeX table sorted by Prompter, with alternating grey shades for models,
    column D10 shifted to appear after D9, and Average column moved to the end.
    Only includes specified Prompters in a defined order.

    Args:
        df (pd.DataFrame): Input DataFrame.
        grey_shades (Dict[str, str]): Mapping of models to their corresponding LaTeX row color.

    Returns:
        str: LaTeX table as a string.
    """
    # Define the desired order of Prompters
    desired_prompters = ["1 center PPV", '1PPV', "2 center PPV", '2PPV', "3 center PPV", '3PPV', "5 center PPV", '5PPV' "10 center PPV", '10PPV', "3D Box"]

    # Filter and reorder Prompters
    df = df[df['Prompter'].isin(desired_prompters)]
    df['Prompter'] = pd.Categorical(df['Prompter'], categories=desired_prompters, ordered=True)
    df = df.sort_values(by=['Prompter', 'Model'])

    # Move D10 to be after D9 and Average to the end
    cols = list(df.columns)
    d10_index = cols.index('D10')
    d9_index = cols.index('D9')
    average_index = cols.index('Average')

    cols.insert(d9_index + 1, cols.pop(d10_index))  # Move D10 after D9
    cols.append(cols.pop(average_index-1))
    df = df[cols]

    # Generate LaTeX table
    latex_rows = ['\\begin{tabular}{llllrrrrrrrrrr}', '\\toprule']
    latex_rows.append('Prompter & Model & Interactions & ' + ' & '.join(df.columns[3:]) + ' \\\\')
    latex_rows.append('\\midrule')

    current_model = None
    for _, row in df.iterrows():
        # Add row color if model changes
        if row['Model'] != current_model:
            latex_rows.append(grey_shades.get(row['Model'], ''))
            current_model = row['Model']
        # Format row values
        row_values = ' & '.join([str(row[col]) if pd.notna(row[col]) else 'NaN' for col in df.columns])
        latex_rows.append(f'{row_values} \\\\')

    latex_rows.append('\\bottomrule')
    latex_rows.append('\\end{tabular}')
    return '\n'.join(latex_rows)
def prepare_prompter_data(df: pd.DataFrame, realistic_prompters: List[str]) -> pd.DataFrame:
    """Prepares data for realistic prompters."""
    pivot_df = df.pivot_table(index=['Model', 'Prompter', 'Interactions'], columns='Dataset', values='Average_Dice').reset_index()
    return pivot_df[pivot_df['Prompter'].isin(realistic_prompters)]

def calculate_average_and_sort(df: pd.DataFrame, dataset_columns: List[str], prompter_order: List[str]) -> pd.DataFrame:
    """Calculates averages, sorts datasets, and applies prompter order."""
    df['Average'] = df[dataset_columns].mean(axis=1)
    df = df[['Model', 'Prompter', 'Interactions'] + dataset_columns + ['Average']].round(2)
    df['Prompter_Order'] = df['Prompter'].map({p: i for i, p in enumerate(prompter_order)})
    return df.sort_values(by=['Prompter_Order', 'Model', 'Interactions']).drop(columns=['Prompter_Order']).reset_index(drop=True)

def plot_barplot(df, selected_models, title, save_path, models_colors, x_labels, show_plot=True):
    """
    Plots average Dice score with precise control of bar spacing, centering the first bar without duplication.
    """
    # Map model names to their corresponding colors
    color_palette = [models_colors[model] for model in selected_models]

    # Unique x-labels
    x_positions = np.arange(len(x_labels))
    bar_width = 0.25  # Width of each bar
    spacing = 0.05  # Minimal spacing between triplets

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(selected_models):
        # Filter data for the current model and reorder based on x_labels
        model_data = df[df['Model'] == model]
        model_data = model_data.set_index('Prompter').reindex(x_labels).reset_index()

        # Generate x offsets for the current model
        x_offsets = x_positions + (i - len(selected_models) / 2) * bar_width + spacing * (i - 1)

        # Center the first bar ("3D Box")
        if model == selected_models[0]:  # Only adjust for the first bar
            x_offsets[0] = x_positions[0]  # Center the first group's bar

        # Plot the bars for the current model
        plt.bar(
            x_offsets,
            model_data['Average'],
            width=bar_width,
            label=model,
            color=color_palette[i]
        )

    # Add x-axis labels and ticks
    plt.xticks(x_positions, [label.replace("center ", "") for label in x_labels], fontsize=25)
    plt.ylabel('Average Dice Score', size=25)
    plt.title(title, size=30)
    plt.ylim([0, 85])
    plt.yticks(fontsize=20)
    plt.gca().set_facecolor('#f0f0f0')  # Light grey background

    # Add legend in the upper right
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()

    # Save and show plot
    plt.savefig(save_path)
    if show_plot:
        plt.show()
# ========== Main Script ==========
if __name__ == '__main__':
    # Load and preprocess data
    df: pd.DataFrame = pd.read_pickle('/home/c306h/cluster-data/intra_bench/results/processed_res.pkl')

    # Define prompter categories
    realistic_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'Box3DVolumePrompter', 'OnePoints3DVolumePrompter', 'TwoPoints3DVolumePrompter', 'ThreePoints3DVolumePrompter',
            'FivePoints3DVolumePrompter', 'TenPoints3DVolumePrompter',
            'OnePointsFromCenterCropped3DVolumePrompter', 'TwoPointsFromCenterCropped3DVolumePrompter',
            'ThreePointsFromCenterCropped3DVolumePrompter', 'FivePointsFromCenterCropped3DVolumePrompter',
            'TenPointsFromCenterCropped3DVolumePrompter',
        ]
    ]


    threeD_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'Box3DVolumePrompter',
            'OnePointsFromCenterCropped3DVolumePrompter', 'TwoPointsFromCenterCropped3DVolumePrompter',
            'ThreePointsFromCenterCropped3DVolumePrompter', 'FivePointsFromCenterCropped3DVolumePrompter',
            'TenPointsFromCenterCropped3DVolumePrompter',
        ]
    ]


    # Prepare and sort data
    prepared_df: pd.DataFrame = prepare_prompter_data(df, realistic_prompters)
    dataset_columns: List[str] = [col for col in prepared_df.columns if col.startswith('D')]
    sorted_df: pd.DataFrame = calculate_average_and_sort(prepared_df, dataset_columns, realistic_prompters)

    # Generate LaTeX output
    latex_table: str = generate_latex(sorted_df, GREY_SHADES)
    print(latex_table)

    # Plotting examples

    threeD_df = sorted_df[sorted_df['Prompter'].isin(threeD_prompters)]
    selected_models_3D: List[str] = [' SegVol', 'SamMed 3D', 'SamMed 3D Turbo',]

    plot_barplot(
        df=threeD_df[threeD_df['Model'].isin(selected_models_3D)],
        selected_models=selected_models_3D,
        title='3D Models',
        save_path='/home/c306h/PAPER_VISUALS/INTRABENCH/res/static_3D_correct_colors.png',
        models_colors=models_colors,  # Ensure this contains the correct mapping
        x_labels=threeD_prompters,
        show_plot=True
    )