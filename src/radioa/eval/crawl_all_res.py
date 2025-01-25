from batchgenerators.utilities.file_and_folder_operations import *
import glob
import pandas as pd
import pickle
from vispy.visuals.line.line import joins


models_colors = {
    "SAM": "#8cc5e3",
    "SAM2": "#1a80bb",
    "SamMed 2D": "#bdd373",  #
    "MedSam": "#7f7f7f",  #
    "SamMed 3D": "#d8a6a6",  # "#d9f0a3",
    "SamMed 3D Turbo": "#a00000",  # "#78c679",
    "SegVol": "#59a89c",  #  "#006837",
}
short_d_names = {
    'Dataset201_MS_Flair_instances': 'D1',
    'Dataset209_hanseg_mr_oar': 'D2',
    'Dataset501_hntsmrg_pre_primarytumor': 'D3',
    'Dataset930_RIDER_LungCT': 'D4',
    'Dataset911_LNQ_instances': 'D5',
    'Dataset912_colorectal_livermets': 'D6',
    'Dataset913_adrenal_acc_ki67': 'D7',
    'Dataset600_pengwin': 'D9',
    'Dataset920_hcc_tace_liver': 'D11',
    'Dataset921_hcc_tace_lesion': 'D12',
    'Dataset651_segrap': 'D10',

}
short_m_names = {'medsam': 'MedSam', 'sammed3d': 'SamMed 3D', 'sammed2d': 'SamMed 2D', 'sam': 'SAM', 'sammed3d_turbo': 'SamMed 3D Turbo', 'sam2': 'SAM2', 'segvol': 'SegVol'}

prompter_names = {'PointPropagationPrompter': ['5P Prop', '7'],
                  'Alternating5PointsPer2DSlicePrompter': ['5$\pm$PPS', '5X'],
                  'TenFGPointsPer2DSlicePrompter': ['10PPS', '10X'],
                  'BoxPropagationPrompter': ['B Prop', '4'],
                  'OnePoints3DVolumePrompter': ['1PPV', '1'],
                  'FiveFGPointsPer2DSlicePrompter': ['5PPS', '5X'],
                  'Alternating3PointsPer2DSlicePrompter': ['3$\pm$PPS', '3X' ],
                  'Alternating2PointsPer2DSlicePrompter': ['2$\pm$PPS', '2X'],
                  'OneFGPointsPer2DSlicePrompter': ['1PPS', '1X' ],
                  'ThreeFGPointsPer2DSlicePrompter': ['3PPS', '3X' ],
                  'CenterPointPrompter': ['1 Center P', '1'],
                  'TwoFGPointsPer2DSlicePrompter': ['2PPS', '2X'],
                  'BoxInterpolationPrompter': ['3B Inter', '6'],
                  'BoxPer2DSlicePrompter': ['Box PS', '2X'],
                  'Alternating10PointsPer2DSlicePrompter': ['10$\pm$PPS', '10X' ],
                  'BoxPer2dSliceFrom3DBoxPrompter': ['from 3D Box', '3'],
                  'PointInterpolationPrompter': ['P Inter', '3'],
                  'ThreePointInterpolationPrompter': ['P Inter', '3'],
                  'ThreeFGPointsPer2DSlicePrompter': ['3PPS', '3X' ],
                  'FivePointInterpolationPrompter':['5P Inter', '5'],
                  'TenPointInterpolationPrompter': ['10P Inter', '10'],
                  'ThreeBoxInterpolationPrompter': ['3B Inter', '6'],
                  'FiveBoxInterpolationPrompter': ['5B Inter', '10'],
                  'TenBoxInterpolationPrompter': ['10B Inter', '20'],
                  'TwoPoints3DVolumePrompter': ['2PPV', '2'],
                  'ThreePoints3DVolumePrompter':['3PPV', '3'],
                  'FivePoints3DVolumePrompter': ['5PPV', '5'],
                  'TenPoints3DVolumePrompter': ['10PPV', '10'],
                  'OnePointsFromCenterCropped3DVolumePrompter': ['1 center PPV', '1'],
                  'TwoPointsFromCenterCropped3DVolumePrompter': ['2 center PPV', '2'],
                  'ThreePointsFromCenterCropped3DVolumePrompter': ['3 center PPV', '3'],
                  'FivePointsFromCenterCropped3DVolumePrompter': ['5 center PPV', '5'],
                  'TenPointsFromCenterCropped3DVolumePrompter': ['10 center PPV', '10'],
                  'Box3DVolumePrompter': ['3D Box', '3'],
                  'OnePointPer2DSliceInteractivePrompterNoPrevPoint': ['1PPS + Scribble Refine', '1x/3'],
                  'OnePointPer2DSliceInteractivePrompterWithPrevPoint': ['1PPS + Scribble Refine*', '1X/3'],
                  'twoD1PointUnrealisticInteractivePrompterWithPrevPoint': ['1PPS + 1PPS Refine*', '1X/1X'],
                  'twoD1PointUnrealisticInteractivePrompterNoPrevPoint': ['1PPS + 1PPS Refine', '1X/1X'],
                  'PointInterpolationInteractivePrompterNoPrevPoint': ['5P Inter + Scribble  Refine', '5/3'],
                  'PointInterpolationInteractivePrompterWithPrevPoint': ['5P Inter + Scribble  Refine*', '5/3'],
                  'PointPropagationInteractivePrompterWithPrevPoint': ['P Prop + Scribble Refine*', '7/3'],
                  'PointPropagationInteractivePrompterNoPrevPoint': ['P Prop + Scribble Refine', '7/3'],
                  'threeDCroppedInteractivePrompterNoPrevPoint': ['1PPV + 1 PPV Refine', '1/1'],
                  'threeDCroppedInteractivePrompterWithPrevPoint': ['1PPV + 1 PPV Refine*', '1/1'],
                  'threeDCroppedFromCenterInteractivePrompterNoPrevPoint': ['1 center PPV +  1 PPV Refine', '1/1'],
                  'threeDCroppedFromCenterInteractivePrompterWithPrevPoint': ['1 center PPV + 1 PPV Refine*', '1/1'],
                  'threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint': ['1 center PPV + Scribble Refine', '1/3'],
                  'threeDCroppedFromCenterAnd2dAlgoInteractivePrompterWithPrevPoint': ['1 center PPV + Scribble Refine*', '1/3'],
                  'BoxInterpolationInteractivePrompterNoPrevPoint': ['3B Inter + Scribble Refine', '6/3']
                  }


if __name__ == '__main__':
    inpath = '/home/c306h/cluster-data/intra_bench/results/'
    if not isfile(join(inpath, "crawled_res.pkl")):
        if not isfile(join(inpath, 'json_files_les.pkl')):
            json_files_les = glob.glob(join(inpath, '**/*results.json'), recursive=True)
            json_files_sem = glob.glob(join(inpath, '**/*agg.json'), recursive=True)

            with open(join(inpath, 'json_files_les.pkl'), 'wb') as file:
                pickle.dump(json_files_les, file)
            with open(join(inpath, 'json_files_sem.pkl'), 'wb') as file:
                pickle.dump(json_files_sem, file)

        else:
            with open(join(inpath, 'json_files_les.pkl'), 'rb') as file:
                json_files_les = pickle.load(file)
            with open(join(inpath, 'json_files_sem.pkl'), 'rb') as file:
                json_files_sem = pickle.load(file)

        data = []
        #instance eval
        for json_file in json_files_les:

            #lesion instance results
            if json_file.endswith('aggregated_lwcw_results.json'):
                #static lesion results
                if not json_file.split('/')[-2].startswith('iter'):
                    res_json = load_json(json_file)
                    dice_value = res_json['1']['lw_dice_mean']
                    if dice_value is not None:
                        # Append the relevant data to the list
                        data.append({
                            'Iteration': '0',
                            'Dataset':  short_d_names[json_file.split('/')[-5]],
                            'Model':  short_m_names[json_file.split('/')[-4]],
                            'Prompter': prompter_names[json_file.split('/')[-3]][0],
                            'Interactions': prompter_names[json_file.split('/')[-3]][1],
                            'Dice': dice_value
                        })
                    else:
                        print('Dice None: ', json_file)
                #iterative lesion results
                if json_file.split('/')[-2].startswith('iter'):
                    res_json = load_json(json_file)
                    dice_value = res_json['1']['lw_dice_mean']
                    if dice_value is not None:
                        # Append the relevant data to the list
                        data.append({
                            'Dataset':  short_d_names[json_file.split('/')[-6]],
                            'Model':  short_m_names[json_file.split('/')[-5]],
                            'Prompter': prompter_names[json_file.split('/')[-4]][0],
                            'Interactions': prompter_names[json_file.split('/')[-4]][1],
                            'Dice': dice_value,
                            'Iteration': json_file.split('/')[-2][-1],
                        })
                    else:
                        print('Dice None: ', json_file)

            # semantic eval
        for json_file in json_files_sem:
            #semantic
            if json_file.endswith('_agg.json'):
                #static lesion results
                if not json_file.split('/')[-2].startswith('iter'):
                    res_json = load_json(json_file)
                    dice_value = res_json['1'][0]['dice']['mean']
                    if dice_value is not None:
                        # Append the relevant data to the list
                        data.append({
                            'Iteration': '0',
                            'Dataset':  short_d_names[json_file.split('/')[-4]],
                            'Model':  short_m_names[json_file.split('/')[-3]],
                            'Prompter': prompter_names[json_file.split('/')[-2]][0],
                            'Interactions': prompter_names[json_file.split('/')[-2]][1],
                            'Dice': dice_value,
                            'Class':  json_file.split('/')[-1][1:3]
                        })
                    else:
                        print('Dice None: ', json_file)
                #iterative lesion results
                if json_file.split('/')[-2].startswith('iter'):
                    res_json = load_json(json_file)
                    dice_value = res_json['1'][0]['dice']['mean']
                    if dice_value is not None:
                        # Append the relevant data to the list
                        data.append({
                            'Dataset':  short_d_names[json_file.split('/')[-5]],
                            'Model':  short_m_names[json_file.split('/')[-4]],
                            'Prompter': prompter_names[json_file.split('/')[-3]][0],
                            'Interactions': prompter_names[json_file.split('/')[-3]][1],
                            'Dice': dice_value,
                            'Iteration': json_file.split('/')[-2][-6],
                            'Class':  json_file.split('/')[-1][1:3]
                        })
                    else:
                        print('Dice None: ', json_file)

        df = pd.DataFrame(data, columns=['Dataset', 'Model', 'Prompter', 'Dice', 'Class', 'Iteration', 'Interactions'])
        df.to_pickle(join(inpath, "crawled_res.pkl"))
    else:
        df = load_pickle(join(inpath, "crawled_res.pkl"))
        # print(df.head())

    #process dataframe
    df['Dice'] = df['Dice'] * 100

    # Merge D11 and D12
    filtered_df = df[df['Dataset'].isin(['D11', 'D12'])]

    # Change Class to '02' where Dataset is 'D12' and Class is '01'
    filtered_df.loc[(filtered_df['Dataset'] == 'D12') & (filtered_df['Class'] == '01'), 'Class'] = '02'
    filtered_df.loc[filtered_df['Dataset'].isin(['D11', 'D12']), 'Dataset'] = 'D8'
    # Remove D11 and D12 from the original dataframe
    df_cleaned = df[~df['Dataset'].isin(['D11', 'D12'])]
    processed_df = pd.concat([df_cleaned, filtered_df], ignore_index=True)
    df.to_pickle(join(inpath, 'merged_res.pkl'))

    #average over classes
    noclasses_df = processed_df.groupby(['Dataset', 'Model', 'Prompter', 'Iteration', 'Interactions'], as_index=False).agg({'Dice': 'mean'})
    noclasses_df = noclasses_df.rename(columns={'Dice': 'Average_Dice'})
    noclasses_df.to_pickle(join(inpath, 'processed_res.pkl'))


