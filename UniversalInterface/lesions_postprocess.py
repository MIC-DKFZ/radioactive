import os
import nibabel as nib
import numpy as np

# Merge segmentations into one binary mask
results_dir = '/home/t722s/Desktop/ExperimentResults_lesions/sammed2d_infer_max_sub_20240726_1714/'
seg_dirs = [os.path.join(results_dir, f) for f in os.listdir(results_dir)]
seg_dirs = [d for d in seg_dirs if os.path.isdir(d)] # Subset to folders

for seg_dir in seg_dirs:
    segs = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]

    summed_image = None

    for seg_path in segs:
        # Load the NIfTI file using nibabel
        img = nib.load(seg_path)
        img_data = img.get_fdata()
        
        if summed_image is None:
            # Initialize the summed_image with the first image data
            summed_image = img_data.copy()
        else:
            # Check if the current image has the same shape as the summed_image
            if img_data.shape != summed_image.shape:
                raise ValueError("All images must have the same dimensions")
            # Add the current image data to the summed_image
            summed_image += img_data

    merged_image = np.where(summed_image>0, 1, 0)

    merged_nifti = nib.Nifti1Image(merged_image, affine=img.affine, header=img.header)
    merged_nifti.to_filename(os.path.join(seg_dir, 'merged_seg.nii.gz'))

