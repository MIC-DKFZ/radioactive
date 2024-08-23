import os
import nibabel as nib
import json
import numpy as np
from lesions_experiments_runner import get_imgs_gts

# Obtain img, gt paths
def preprocess_save_instances(dataset_dir):
    dataset_dir = '/home/t722s/Desktop/Datasets/melanoma_HD_sub/'
    imgs_gts = get_imgs_gts(dataset_dir)

    instances_dict = {split: 
                            {
                                os.path.basename(gt): []
                                for _, gt  in imgs_gts[split]
                            }
                        for  split in ['Tr', 'Ts']
                        }

    for split in ['Tr', 'Ts']:
        for _, gt_path in imgs_gts[split]:
            # Load in gt, find instances
            gt = nib.load(gt_path).get_fdata()
            instances = list(np.unique(gt))
            # Exclude background and convert to json serializable integer type. 
            instances_dict[split][os.path.basename(gt_path)] = {str(int(i)): int(i) for i in instances if i!=0} # Redundant dictionaries are used for similarity to organ dataset.json files

    instances_save_path = os.path.join(dataset_dir, 'instances.json')
    with open(instances_save_path, 'w') as f:
        json.dump(instances_dict, f, indent = 4)

    print(f'Instances per gt found and saved in {instances_save_path}')

if __name__ == '__main__':
    dataset_dir = '/home/t722s/Desktop/Datasets/melanoma_HD_sub/'

    preprocess_save_instances(dataset_dir)