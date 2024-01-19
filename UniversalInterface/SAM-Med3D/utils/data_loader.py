from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import torch
import numpy as np
import os
import pickle
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
import einops as E


class Dataset_Union_ALL(Dataset): 
    def __init__(self, paths, points_path, dim, label = 1, mode='train', data_type='Tr', image_size=128, 
                 transform=None, threshold=500,
                 split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.data_type = data_type
        self.split_num=split_num
        self.split_idx=split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc

        # Added by Tim:
        self.label = label # Current implementation will need a different dataset/dataloader per foreground region since the patch we're taking crops based upon the foreground points selected, which in turn depend on the label.
        self.dim = dim

        with open(points_path, 'rb') as f:
            self.points_dict = pickle.load(f)
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        # Added by Tim
        sitk_label = sitk.ReadImage(self.label_paths[index]) # REMOVE later; use for DICE calculation for now
        tio_image = tio.ScalarImage.from_sitk(sitk.ReadImage(self.image_paths[index]))
        
        # Load in points for this image and change to a mask to be usable by torchio
        points = self.points_dict[os.path.basename(self.image_paths[index])][self.label][str(self.dim) + 'D']
        points_mask = np.zeros(shape = (tio_image.shape[3], tio_image.shape[2], tio_image.shape[1])) # Skip the color channel for now; reintroduce in label_map definition. Reverse order since sitk uses WHD while numpy uses DHW 
        points_mask[*points.T] = 1
        points_mask = E.rearrange(points_mask, pattern = 'x y z -> z y x') # Rearrange back to sitk WHD
        points_mask = tio.LabelMap(tensor = torch.from_numpy(points_mask).float().unsqueeze(0), affine = tio_image.affine)

        subject = tio.Subject(
            image = tio_image,
            points_mask = points_mask,
            label = tio.LabelMap.from_sitk(sitk_label)
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])
 
        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index] # Later don't return label data

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f'labels{self.data_type}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    self.image_paths.append(label_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)

class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, path):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for dt in ["Tr", "Val", "Ts"]:
            d = os.path.join(path, f'labels{dt}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]
                    label_path = os.path.join(path, f'labels{dt}', f'{base}.nii.gz') 
                    self.image_paths.append(label_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)
        self.image_paths = self.image_paths[self.split_idx::self.split_num]
        self.label_paths = self.label_paths[self.split_idx::self.split_num]




class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# class Test_Single(Dataset):  #OUTDATED
#     def __init__(self, paths, image_size=128, transform=None, threshold=500):
#         self.paths = paths

#         self._set_file_paths(self.paths)
#         self.image_size = image_size
#         self.transform = transform
#         self.threshold = threshold
    
#     def __len__(self):
#         return len(self.label_paths)

#     def __getitem__(self, index):

#         sitk_image = sitk.ReadImage(self.image_paths[index])
#         sitk_label = sitk.ReadImage(self.label_paths[index])

#         if sitk_image.GetOrigin() != sitk_label.GetOrigin():
#             sitk_image.SetOrigin(sitk_label.GetOrigin())
#         if sitk_image.GetDirection() != sitk_label.GetDirection():
#             sitk_image.SetDirection(sitk_label.GetDirection())

#         subject = tio.Subject(
#             image = tio.ScalarImage.from_sitk(sitk_image),
#             label = tio.LabelMap.from_sitk(sitk_label),
#         )

#         if '/ct_' in self.image_paths[index]:
#             subject = tio.Clamp(-1000,1000)(subject)

#         if self.transform:
#             try:
#                 subject = self.transform(subject)
#             except:
#                 print(self.image_paths[index])


#         if subject.label.data.sum() <= self.threshold:
#             return self.__getitem__(np.random.randint(self.__len__()))
        

#         return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
    
#     def _set_file_paths(self, paths):
#         self.image_paths = []
#         self.label_paths = []

#         self.image_paths.append(paths)
#         self.label_paths.append(paths.replace('images', 'labels'))



if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL(
        paths=['/cpfs01/shared/gmai/medical_preprocessed/3d/iseg/ori_totalseg_two_class/liver/Totalsegmentator_dataset_ct/',], 
        data_type='Ts', 
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label', target_shape=(128,128,128)),
        ]), 
        threshold=0)

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    for i,j,n in test_dataloader:
        # print(i.shape)
        # print(j.shape)
        # print(n)
        continue
