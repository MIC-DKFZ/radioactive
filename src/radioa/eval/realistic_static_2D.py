import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from radioa.eval.crawl_all_res import prompter_names, models_colors

# Define grey shades for LaTeX table background
GREY_SHADES: Dict[str, str] = {
    'SAM': '\\rowcolor[gray]{1}',
    'SAM2': '\\rowcolor[gray]{0.95}',
    'SamMed 2D': '\\rowcolor[gray]{0.9}',
    'ScribblePrompt': '\\rowcolor[gray]{0.85}',
    'MedSam': '\\rowcolor[gray]{0.8}',
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
    desired_prompters = ['3P Inter', '5P Inter','10P Inter', '5P Prop', 'B Prop', 'from 3D Box', '3B Inter','5B Inter',  '10B Inter']

    # Filter and reorder Prompters
    df = df[df['Prompter'].isin(desired_prompters)]
    df['Prompter'] = pd.Categorical(df['Prompter'], categories=desired_prompters, ordered=True)
    model_order = ['SAM', 'SAM2', 'SamMed 2D', 'ScribblePrompt', 'MedSam']
    df['Model_Order'] = df['Model'].map({m: i for i, m in enumerate(model_order)})
    df = df.sort_values(by=['Prompter', 'Model_Order']).drop(columns=['Model_Order'])

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

def calculate_average_and_sort(df: pd.DataFrame, dataset_columns: List[str], prompter_order: List[str], model_order: List[str]) -> pd.DataFrame:
    """Calculates averages, sorts datasets, and applies prompter and model order."""
    df['Average'] = df[dataset_columns].mean(axis=1)
    df = df[['Model', 'Prompter', 'Interactions'] + dataset_columns + ['Average']].round(2)

    df['Prompter_Order'] = df['Prompter'].map({p: i for i, p in enumerate(prompter_order)})
    df['Model_Order'] = df['Model'].map({m: i for i, m in enumerate(model_order)})

    df = df.sort_values(by=['Prompter_Order', 'Model_Order', 'Interactions'])

    return df.drop(columns=['Prompter_Order', 'Model_Order']).reset_index(drop=True)

def plot_average_dice(df: pd.DataFrame, selected_models: List[str], title: str, save_path: str,
                      color_palette: List[str], x_labels_order: List[str] = None, show_plot: bool = True) -> None:
    """Plots average Dice score for the given models and optionally displays the plot."""

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x='Prompter', y='Average', hue='Model', palette=color_palette, order=x_labels_order)
    plt.gca().set_facecolor('#f0f0f0')  # Light grey background
    plt.title(title, size=20)
    plt.xlabel('')
    plt.ylabel('Average Dice Score', size=20)
    plt.ylim([0, 85])
    plt.yticks(size=15)
    plt.xticks(rotation=0, size=15)
    sns.set_theme(style="whitegrid")
    plt.gca().set_axisbelow(True)
    plt.grid(True, color='white')

    # Get the current handles and labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    #
    # # Define the desired order of legend labels
    #
    # if 'MedSam' in labels:
    #     legend_order = ['ScribblePrompt', 'SamMed 2D', 'MedSam','SAM', 'SAM2']
    # else:
    #     legend_order = ['ScribblePrompt' ,'SamMed 2D', 'SAM', 'SAM2']
    #
    # # Reorder handles and labels according to legend_order
    # ordered_handles = [handles[labels.index(label)] for label in legend_order]
    # ordered_labels = [label for label in legend_order]
    #
    # # Set the legend with the new order
    # plt.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=20)



    plt.legend(loc='upper left', fontsize=17)
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
            'ThreePointInterpolationPrompter', 'FivePointInterpolationPrompter', 'TenPointInterpolationPrompter',
            'PointPropagationPrompter', 'BoxPropagationPrompter', 'BoxPer2dSliceFrom3DBoxPrompter','ThreeBoxInterpolationPrompter',
            'FiveBoxInterpolationPrompter', 'TenBoxInterpolationPrompter',
        ]
    ]

    point_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'ThreePointInterpolationPrompter', 'FivePointInterpolationPrompter', 'TenPointInterpolationPrompter',
            'PointPropagationPrompter',
        ]
    ]

    box_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'BoxPropagationPrompter', 'ThreeBoxInterpolationPrompter',
            'FiveBoxInterpolationPrompter', 'TenBoxInterpolationPrompter',
        ]
    ]

    # Prepare and sort data
    prepared_df: pd.DataFrame = prepare_prompter_data(df, realistic_prompters)
    dataset_columns: List[str] = [col for col in prepared_df.columns if col.startswith('D')]
    model_order = ['SAM', 'SAM2', 'SamMed 2D', 'ScribblePrompt', 'MedSam']
    sorted_df: pd.DataFrame = calculate_average_and_sort(prepared_df, dataset_columns, realistic_prompters, model_order=model_order)

    # Generate LaTeX output
    latex_table: str = generate_latex(sorted_df, GREY_SHADES)
    print(latex_table)

    # Plotting examples
    point_df = sorted_df[sorted_df['Prompter'].isin(point_prompters)]
    selected_models_2D_points: List[str] = ['SAM', 'SAM2', 'SamMed 2D', 'ScribblePrompt',]
    plot_average_dice(
        point_df[point_df['Model'].isin(selected_models_2D_points)],
        selected_models_2D_points,
        title='Realistic Point Prompts 2D Models',
        save_path='/home/c306h/PAPER_VISUALS/INTRABENCH/res/barplot_realistic_points_static_prompter.png',
        color_palette=[models_colors[m] for m in selected_models_2D_points],
        show_plot=True  # Ensure the plot is displayed
    )
    box_df = sorted_df[sorted_df['Prompter'].isin(box_prompters)]
    selected_models_2D_box: List[str] =  ['MedSam','SAM', 'SAM2', 'SamMed 2D', 'ScribblePrompt',]
    box_df['Model_Order'] = box_df['Model'].map({m: i for i, m in enumerate(selected_models_2D_box)})
    box_df = box_df.sort_values(by=['Model_Order',]).drop(columns=['Model_Order'])

    plot_average_dice(
        box_df[box_df['Model'].isin(selected_models_2D_box)],
        selected_models_2D_box,
        title='Realistic Box Prompts 2D Models',
        save_path='/home/c306h/PAPER_VISUALS/INTRABENCH/res/barplot_realistic_boxes_static_prompter.png',
        color_palette=[models_colors[m] for m in selected_models_2D_box],
        show_plot=True  # Ensure the plot is displayed
    )
