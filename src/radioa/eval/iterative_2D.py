import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from radioa.eval.crawl_all_res import prompter_names, models_colors

# Define grey shades for LaTeX table background
GREY_SHADES: Dict[str, str] = {
    'MedSam': '\\rowcolor[gray]{1}',
    'SAM': '\\rowcolor[gray]{0.95}',
    'SAM2': '\\rowcolor[gray]{0.9}',
    'SamMed 2D': '\\rowcolor[gray]{0.85}',
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
    desired_prompters = ['1PPS + 1PPS Refine', '1PPS + 1PPS Refine*',
                         '1PPS + Scribble Refine', '1PPS + Scribble Refine*',
                         '3B Inter + Scribble Refine', '5P Inter + Scribble  Refine']

    # Filter and reorder Prompters
    df = df[df['Prompter'].isin(desired_prompters)]
    df['Prompter'] = pd.Categorical(df['Prompter'], categories=desired_prompters, ordered=True)
    df = df.sort_values(by=['Prompter', 'Model', 'Iteration'])

    # Move D10 to be after D9 and Average to the end
    cols = list(df.columns)
    d10_index = cols.index('D10')
    d9_index = cols.index('D9')
    average_index = cols.index('Average')

    cols.insert(d9_index + 1, cols.pop(d10_index))  # Move D10 after D9
    cols.append(cols.pop(average_index-1))
    df = df[cols]

    # Generate LaTeX table
    latex_rows = ['\\begin{tabular}{llllrrrrrrrrrrr}', '\\toprule']
    latex_rows.append('Model & Promter & Interactions & Iteration &' + ' & '.join(df.columns[4:]) + ' \\\\')
    latex_rows.append('\\midrule')

    current_model = None
    for _, row in df.iterrows():
        # Add row color if model changes
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
    pivot_df = df.pivot_table(index=['Model', 'Prompter', 'Interactions', 'Iteration'], columns='Dataset', values='Average_Dice').reset_index()
    return pivot_df[pivot_df['Prompter'].isin(realistic_prompters)]

def calculate_average_and_sort(df: pd.DataFrame, dataset_columns: List[str], prompter_order: List[str]) -> pd.DataFrame:
    """Calculates averages, sorts datasets, and applies prompter order."""
    df['Average'] = df[dataset_columns].mean(axis=1)
    df = df[['Model', 'Prompter', 'Interactions', 'Iteration'] + dataset_columns + ['Average']].round(2)
    df['Prompter_Order'] = df['Prompter'].map({p: i for i, p in enumerate(prompter_order)})
    return df.sort_values(by=['Prompter_Order', 'Model', 'Iteration']).drop(columns=['Prompter_Order']).reset_index(drop=True)

def plot_average_dice(df: pd.DataFrame, selected_models: List[str], title: str, save_path: str,
                      color_palette: List[str], x_labels_order: List[str] = None, show_plot: bool = True) -> None:
    """Plots average Dice score for the given models and optionally displays the plot."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Prompter', y='Average', hue='Model', palette=color_palette, order=x_labels_order)
    plt.gca().set_facecolor('#f0f0f0')  # Light grey background
    plt.title(title, size=30)
    plt.xlabel('')
    plt.ylabel('Average Dice Score', size=25)
    plt.ylim([0.01, 85])
    plt.yticks(size=20)
    plt.xticks(rotation=0, size=25)
    plt.legend(loc='upper left', fontsize=20)
    plt.tight_layout()
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
            'twoD1PointUnrealisticInteractivePrompterNoPrevPoint', 'twoD1PointUnrealisticInteractivePrompterWithPrevPoint',
            'OnePointPer2DSliceInteractivePrompterNoPrevPoint','OnePointPer2DSliceInteractivePrompterWithPrevPoint',
            'BoxInterpolationInteractivePrompterNoPrevPoint',
            'PointPropagationInteractivePrompterNoPrevPoint', 'PointPropagationInteractivePrompterWithPrevPoint',
            'PointInterpolationInteractivePrompterWithPrevPoint', 'PointInterpolationInteractivePrompterNoPrevPoint'
        ]
    ]

    final_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'twoD1PointUnrealisticInteractivePrompterWithPrevPoint',
             'OnePointPer2DSliceInteractivePrompterWithPrevPoint',
        ]
    ]


    # Prepare and sort data
    prepared_df: pd.DataFrame = prepare_prompter_data(df, realistic_prompters)
    dataset_columns: List[str] = [col for col in prepared_df.columns if col.startswith('D')]
    sorted_df: pd.DataFrame = calculate_average_and_sort(prepared_df, dataset_columns, realistic_prompters)

    # Generate LaTeX output
    latex_table: str = generate_latex(sorted_df, GREY_SHADES)
    print(latex_table)


    final_df = sorted_df[sorted_df['Prompter'].isin(final_prompters)]




    # Combine 'Model' and 'Prompter' into a new column for unique lines
    final_df['Model_Prompter'] = final_df['Model'] + ": " + final_df['Prompter']

    # Get unique models and prompters
    unique_models = final_df['Model'].unique()
    unique_prompters = final_df['Prompter'].unique()

    # Build a custom palette for each model-prompter combination
    palette = {}
    for model in unique_models:
        for prompter in unique_prompters:
            model_prompter_comb = f'{model}: {prompter}'
            # Directly use the assigned color for the model
            palette[model_prompter_comb] = models_colors[model]

    # Plot using the custom palette
    line_styles = {
        '1PPS + 1PPS Refine*': 'solid',
        '1PPS + Scribble Refine*': 'dashed'
    }

    # Add a new column for line styles based on the prompter
    final_df['Line_Style'] = final_df['Prompter'].map(line_styles)

    plt.figure(figsize=(10, 8))

    # Iterate through unique line styles and plot separately
    for line_style, group_data in final_df.groupby('Line_Style'):
        sns.lineplot(
            data=group_data,
            x='Iteration',
            y='Average',
            hue='Model_Prompter',
            palette=palette,
            linestyle=line_style,
            markersize=10,
            linewidth=3,
            marker="o"
        )

    # Add labels and title
    plt.xlabel('Iteration', size=20)
    plt.ylim([0, 62])
    plt.title('3D model Refinement', fontsize=20)
    plt.gca().set_facecolor('#f0f0f0')
    plt.ylabel('Average Dice Score', size=20)
    plt.xticks(rotation=0, size=15)
    plt.yticks(size=15)
    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()

    # Show the plot
    plt.savefig('/home/c306h/PAPER_VISUALS/INTRABENCH/res/interactiveline2D.png')
    plt.show()