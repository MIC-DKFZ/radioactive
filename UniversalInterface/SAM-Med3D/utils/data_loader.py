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

def _bbox_mask(mask_volume: np.ndarray):
        """Return 6 coordinates of a 3D bounding box from a given mask.

        Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.

        Args:
            mask_volume: 3D NumPy array.
        """  # noqa: B950
        i_any = np.any(mask_volume, axis=(1, 2))
        j_any = np.any(mask_volume, axis=(0, 2))
        k_any = np.any(mask_volume, axis=(0, 1))
        i_min, i_max = np.where(i_any)[0][[0, -1]]
        j_min, j_max = np.where(j_any)[0][[0, -1]]
        k_min, k_max = np.where(k_any)[0][[0, -1]]
        bb_min = np.array([i_min, j_min, k_min])
        bb_max = np.array([i_max, j_max, k_max]) + 1
        return bb_min, bb_max

def getCroppingParams(subject, mask_name, target_shape):
    '''Function to get the cropping and padding parameters used in an apply_transform call of torchio.CropOrPad, which can then be used to invert the transformation later on'''

    mask_data = subject[mask_name].data.bool().numpy()

    subject_shape = subject.spatial_shape
    bb_min, bb_max = _bbox_mask(mask_data[0])
    center_mask = np.mean((bb_min, bb_max), axis=0)
    padding = []
    cropping = []

    for dim in range(3):
        target_dim = target_shape[dim]
        center_dim = center_mask[dim]
        subject_dim = subject_shape[dim]

        center_on_index = not (center_dim % 1)
        target_even = not (target_dim % 2)

        # Approximation when the center cannot be computed exactly
        # The output will be off by half a voxel, but this is just an
        # implementation detail
        if target_even ^ center_on_index:
            center_dim -= 0.5

        begin = center_dim - target_dim / 2
        if begin >= 0:
            crop_ini = begin
            pad_ini = 0
        else:
            crop_ini = 0
            pad_ini = -begin

        end = center_dim + target_dim / 2
        if end <= subject_dim:
            crop_fin = subject_dim - end
            pad_fin = 0
        else:
            crop_fin = 0
            pad_fin = end - subject_dim

        padding.extend([pad_ini, pad_fin])
        cropping.extend([crop_ini, crop_fin])
    
    # Conversion for SimpleITK compatibility
    padding_array = np.asarray(padding, dtype=int)
    cropping_array = np.asarray(cropping, dtype=int)
    if padding_array.any():
        padding_params = tuple(padding_array.tolist())
    else:
        padding_params = None
    if cropping_array.any():
        cropping_params = tuple(cropping_array.tolist())
    else:
        cropping_params = None
    return padding_params, cropping_params  # type: ignore[return-value]


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
        self.label = label # Future implementation will need a different dataset/dataloader per foreground label since the patch we're taking crops based upon the foreground points selected, which in turn depend on the label.
        self.dim = dim

        with open(points_path, 'rb') as f:
            self.points_dict = pickle.load(f)
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        # sitk_label = sitk.ReadImage(self.label_paths[index]) # REMOVE later; use for DICE calculation for now
        # tio_image = tio.ScalarImage.from_sitk(sitk.ReadImage(self.image_paths[index]))
        
        # # Load in points for this image and change to a mask to be usable by torchio for cropping/padding
        # points_list = self.points_dict[os.path.basename(self.image_paths[index])][self.label][str(self.dim) + 'D']
        # points_mask = np.zeros(shape = (tio_image.shape[3], tio_image.shape[2], tio_image.shape[1])) # Skip the color channel for now; reintroduce in label_map definition. Reverse order since sitk uses WHD while numpy uses DHW 
        # points_mask[*points_list.T] = 1
        # points_mask = E.rearrange(points_mask, pattern = 'x y z -> z y x') # Rearrange back to sitk WHD
        # points_mask = tio.LabelMap(tensor = torch.from_numpy(points_mask).float().unsqueeze(0), affine = tio_image.affine)

        # subject = tio.Subject(
        #     image = tio_image,
        #     points_mask = points_mask,
        #     label = tio.LabelMap.from_sitk(sitk_label)
        # )

        # obtain cropping and padding parameters to permit later inversion of the transform
        target_shape = self.transform[1].target_shape # could provide error checking to check that self.transform[1] is indeed a croporpad. 

        label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_paths[index])) # REMOVE later; use for DICE calculation for now
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[index]))
        
        # Load in points for this image and change to a mask to be usable by torchio for cropping/padding
        points_list = self.points_dict[os.path.basename(self.image_paths[index])][self.label][str(self.dim) + 'D']
        points_mask = np.zeros(shape = image.shape) 
        points_mask[*points_list.T] = 1

        subject = tio.Subject(
            image = tio.ScalarImage(tensor = torch.from_numpy(image).permute(2,1,0).unsqueeze(0)), # add channel dimension to everything, and permute to x,y,z orientation
            points_mask = tio.LabelMap(tensor = torch.from_numpy(points_mask).permute(2,1,0).float().unsqueeze(0)),
            label = tio.LabelMap(tensor = torch.from_numpy(label).permute(2,1,0).unsqueeze(0))
        )
        
        pad_crop_params = getCroppingParams(subject, 'label', target_shape)

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])
 
        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), subject.points_mask.data.squeeze(0), torch.tensor(pad_crop_params), self.image_paths[index] # Later don't return label data. Remove channel dimension from points_mask

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
