import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from radioa.eval.crawl_all_res import prompter_names, models_colors



# ========== Utility Functions ==========

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
            'PointInterpolationInteractivePrompterWithPrevPoint', 'PointInterpolationInteractivePrompterNoPrevPoint',
            'BoxInterpolationInteractivePrompterWithPrevBox',   'threeDCroppedFromCenterInteractivePrompterNoPrevPoint',
            'threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint',
        ]
    ]

    final_prompters: List[str] = [
        prompter_names[p][0] for p in [
            'twoD1PointUnrealisticInteractivePrompterWithPrevPoint',
            'OnePointPer2DSliceInteractivePrompterWithPrevPoint',
            'BoxInterpolationInteractivePrompterWithPrevBox', 'threeDCroppedFromCenterInteractivePrompterNoPrevPoint',
            'threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint',
        ]
    ]


    # Prepare and sort data
    prepared_df: pd.DataFrame = prepare_prompter_data(df, realistic_prompters)
    dataset_columns: List[str] = [col for col in prepared_df.columns if col.startswith('D')]
    sorted_df: pd.DataFrame = calculate_average_and_sort(prepared_df, dataset_columns, realistic_prompters)

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
        '1PPS + Scribble Refine*': 'dashed',
        '3B Inter + Scribble Refine*': 'dotted',
        '1 center PPV + Scribble Refine': 'dashed',
        '1 center PPV + 1 PPV Refine': 'solid'
    }

    # Add a new column for line styles based on the prompter
    final_df['Line_Style'] = final_df['Prompter'].map(line_styles)

    plt.figure(figsize=(20, 8))

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
    plt.xlabel('Iteration', size=15)
    plt.ylim([0, 85])
    plt.title('Iterative Model Refinement', fontsize=20)
    plt.gca().set_facecolor('#f0f0f0')
    plt.ylabel('Average Dice Score', size=15)
    plt.xticks(rotation=0, size=15)
    plt.yticks(size=15)

    # Get the current legend handles and labels
    handles, old_labels = plt.gca().get_legend_handles_labels()

    # Define the new order of labels
    label_order = [
        "SAM:\n1PPS + Scribble Refine*",
        "SAM:\n1PPS + 1PPS Refine*",
        "SAM2:\n3B Inter + Scribble Refine*",
        "SAM2:\n1PPS + Scribble Refine*",
        "SAM2:\n1PPS + 1PPS Refine*",
        "SamMed 2D:\n1PPS + Scribble Refine*",
        "SamMed 2D:\n1PPS + 1PPS Refine*",
        "ScribblePrompt:\n1PPS + Scribble Refine*",
        "ScribblePrompt:\n1PPS + 1PPS Refine*",
        "SamMed 3D:\n1 center PPV + Scribble Refine",
        "SamMed 3D:\n1 center PPV + 1 PPV Refine",
        "SamMed 3D Turbo:\n1 center PPV + Scribble Refine",
        "SamMed 3D Turbo:\n1 center PPV + 1 PPV Refine"
    ]

    # Map labels to handles
    label_to_handle = dict(zip(old_labels, handles))

    # Reorder handles according to the new label order
    new_handles = [label_to_handle[label.replace("\n", " ")] for label in label_order]

    # Set the new legend with wider handles
    plt.legend(new_handles, label_order, loc='upper left', bbox_to_anchor=(1, 1),
               fontsize=12, labelspacing=0.74, handletextpad=1.5,
               handlelength=3, handleheight=2)  # Increase handle size


    plt.tight_layout()
    sns.set_theme(style="whitegrid")
    plt.gca().set_axisbelow(True)
    plt.grid(True, color='white')

    # Show the plot
    plt.savefig('/home/c306h/PAPER_VISUALS/INTRABENCH/res/interactivelineall.png')
    plt.show()