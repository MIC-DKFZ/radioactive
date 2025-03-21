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
    'SamMed 3D': '\\rowcolor[gray]{1}',
    'SamMed 3D Turbo': '\\rowcolor[gray]{0.95}',
    'SegVol': '\\rowcolor[gray]{0.9}'
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
    desired_prompters = ['3B Inter', '1 center PPV', '3D Box' ]

    # Filter and reorder Prompters
    df = df[df['Prompter'].isin(desired_prompters)]
    df['Prompter'] = pd.Categorical(df['Prompter'], categories=desired_prompters, ordered=True)
    model_order = ['SAM', 'SAM2', 'SamMed 2D', 'ScribblePrompt', 'MedSam', 'SamMed 3D', 'SamMed 3D Turbo', 'SegVol']
    df['Model_Order'] = df['Model'].map({m: i for i, m in enumerate(model_order)})
    df = df.sort_values(by=['Model_Order', 'Prompter']).drop(columns=['Model_Order'])
    df['Path. Average'] = df[['D1', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8b',]].mean(axis=1).round(2)
    df['Org. Average'] = df[['D2', 'D8a', 'D10']].mean(axis=1).round(2)


    # Generate LaTeX table
    latex_rows = ['\\begin{tabular}{llllrrrrrrrrrrrrrr}', '\\toprule']
    latex_rows.append('Prompter & Model & Interactions & ' + ' & '.join(df.columns[3:]) + ' \\\\')
    latex_rows.append('\\midrule')

    current_model = None
    for _, row in df.iterrows():
        # Add row color if model changes

        latex_rows.append(grey_shades.get(row['Model'], ''))

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


# ========== Main Script ==========
if __name__ == '__main__':
    # Load and preprocess data
    df: pd.DataFrame = pd.read_pickle('/home/c306h/cluster-data/intra_bench/results/processed_res_no_merged.pkl')

    # Define prompter categories
    realistic_prompters: List[str] = [
        prompter_names[p][0] for p in ['ThreeBoxInterpolationPrompter', 'Box3DVolumePrompter', 'OnePointsFromCenterCropped3DVolumePrompter'
                                       ]
    ]

    # Prepare and sort data
    prepared_df: pd.DataFrame = prepare_prompter_data(df, realistic_prompters)
    dataset_columns: List[str] = [col for col in prepared_df.columns if col.startswith('D')]
    model_order = ['SamMed 2D', 'ScribblePrompt', 'MedSam', 'SAM', 'SAM2', 'SamMed 3D', 'SamMed 3D Turbo', 'SegVol']
    sorted_df: pd.DataFrame = calculate_average_and_sort(prepared_df, dataset_columns, realistic_prompters, model_order=model_order)

    # Generate LaTeX output
    latex_table: str = generate_latex(sorted_df, GREY_SHADES)
    print(latex_table)
