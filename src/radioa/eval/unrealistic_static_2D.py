from copy import deepcopy
import matplotlib.lines as mlines
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
    'MedSam': '\\rowcolor[gray]{0.85}',
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
    desired_prompters = ['1PPS', '2PPS','2$\pm$PPS', '3PPS', '3$\pm$PPS', '5PPS','5$\pm$PPS',  '10PPS', '10$\pm$PPS', 'Box PS']

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



# ========== Main Script ==========
if __name__ == '__main__':
    # Load and preprocess data
    df: pd.DataFrame = pd.read_pickle('/home/c306h/cluster-data/intra_bench/results/processed_res.pkl')

    # Define prompter categories
    realistic_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'OneFGPointsPer2DSlicePrompter', 'TwoFGPointsPer2DSlicePrompter', 'ThreeFGPointsPer2DSlicePrompter',
            'FiveFGPointsPer2DSlicePrompter', 'TenFGPointsPer2DSlicePrompter',
            'Alternating2PointsPer2DSlicePrompter', 'Alternating3PointsPer2DSlicePrompter','Alternating5PointsPer2DSlicePrompter',
            'Alternating10PointsPer2DSlicePrompter', 'BoxPer2DSlicePrompter']]
    pos_promter: List[str] = {
        prompter_names[p][0]: c + 1 for c,p in enumerate([
            'OneFGPointsPer2DSlicePrompter', 'TwoFGPointsPer2DSlicePrompter', 'ThreeFGPointsPer2DSlicePrompter',
            'FiveFGPointsPer2DSlicePrompter', 'TenFGPointsPer2DSlicePrompter'])}
    alt_promter: List[str] = {
        prompter_names[p][0]: c + 1 for c,p in enumerate([
            'Alternating2PointsPer2DSlicePrompter', 'Alternating3PointsPer2DSlicePrompter','Alternating5PointsPer2DSlicePrompter',
            'Alternating10PointsPer2DSlicePrompter',])}
    models = ['MedSam', 'SAM', 'SAM2', 'SamMed 2D']


    prepared_df: pd.DataFrame = prepare_prompter_data(df, realistic_prompters)
    dataset_columns: List[str] = [col for col in prepared_df.columns if col.startswith('D')]
    df_sort: pd.DataFrame = calculate_average_and_sort(prepared_df, dataset_columns, realistic_prompters)

    latex_table = generate_latex(df_sort, GREY_SHADES)
    print(latex_table)
    # Define the desired order for the x-axis labels
    x_labels_order = ['1PPS', '2PPS', '3PPS', '5PPS', '10PPS', '2$\pm$PPS', '3$\pm$PPS', '5$\pm$PPS', '10$\pm$PPS']

    # Map the prompter names to their order
    df = deepcopy(df_sort)
    df['Prompter'] = pd.Categorical(df_sort['Prompter'], categories=x_labels_order, ordered=True)
    # Plot positive prompters with solid lines
    for i, model in enumerate(models):
        if model != 'MedSam':
            pos_model_df = df[df['Model'] == model]
            pos_prompter_df = pos_model_df[pos_model_df['Prompter'].isin(pos_promter.keys())]
            sns.lineplot(
                data=pos_prompter_df,
                x='Prompter',
                y='Average',
                marker='o',
                markersize=10,
                linewidth=3,
                color=models_colors[model]
            )

    # Align alternative prompters to the same x positions as their positive counterparts
    alt_to_pos_mapping = {
        '2$\pm$PPS': '2PPS',
        '3$\pm$PPS': '3PPS',
        '5$\pm$PPS': '5PPS',
        '10$\pm$PPS': '10PPS'
    }

    # Plot alternative prompters with dashed lines
    for i, model in enumerate(models):
        if model != 'MedSam':
            alt_model_df = df[df['Model'] == model]
            alt_prompter_df = alt_model_df[alt_model_df['Prompter'].isin(alt_promter.keys())]

            # Adjust x positions for alternative prompters
            alt_prompter_df = alt_prompter_df.copy()
            alt_prompter_df['Mapped_Prompter'] = alt_prompter_df['Prompter'].map(alt_to_pos_mapping)

            sns.lineplot(
                data=alt_prompter_df,
                x='Mapped_Prompter',
                y='Average',
                marker='s',
                linestyle='--',
                color=models_colors[model],
                alpha=0.7,
                markersize=10,
                linewidth=3
            )

    box_ps_data = df_sort[df_sort['Prompter'].isin(['Box PS'])]
    # Calculate average dice values for each model
    models = [model for model in box_ps_data['Model'].unique()]
    average_dice_values = box_ps_data.groupby('Model')['Average'].mean()

    # Add average dice values as star markers at the correct x position
    for model in models:
        if model in average_dice_values.index:
            avg_dice = average_dice_values[model]
            x_position = 1  # Ensure this matches the correct x position
            plt.scatter([x_position], [avg_dice], marker='*', s=200, color=models_colors[model])

    # Customize plot appearance and create custom legend entries
    MedSam = mlines.Line2D([], [], color=models_colors['MedSam'], marker='s', linestyle='None', markersize=10, label='MedSam')
    Sam = mlines.Line2D([], [], color=models_colors['SAM'], marker='s', linestyle='None', markersize=10, label='SAM')
    Sam2 = mlines.Line2D([], [], color=models_colors['SAM2'], marker='s', linestyle='None', markersize=10, label='SAM2')
    SamMed2D = mlines.Line2D([], [], color=models_colors['SamMed 2D'], marker='s', linestyle='None', markersize=10, label='SamMed 2D')

    plt.gca().set_facecolor('#f0f0f0')
    plt.legend(handles=[MedSam, Sam, Sam2, SamMed2D], loc='upper right', fontsize=15)  # Single legend call
    plt.title('BOX & Point Prompts Per Slice (PPS & BPS)', size=15)
    plt.ylabel('Average Dice Score', size=10)
    plt.xlabel('')
    plt.xticks(rotation=0, size=10)
    plt.yticks(size=10)
    plt.ylim([0, 85])
    plt.tight_layout()

    # Display the plot
    plt.savefig('/home/c306h/PAPER_VISUALS/INTRABENCH/res/lineplot_static_prompter_with_stars.png')
    plt.show()