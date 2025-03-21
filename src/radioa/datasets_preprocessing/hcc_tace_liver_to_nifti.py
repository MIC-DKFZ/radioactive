import os
from pydicom import dcmread
import SimpleITK as sitk
import numpy as np


IN_DIR = "hcc_tace_seg"
OUT_DIR = "hcc_tace_nifti"

if __name__ == "__main__":
    patients = os.listdir(IN_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labelsTr"), exist_ok=True)

    for patient in sorted(patients):
        timepoint = os.listdir(os.path.join(IN_DIR, patient))
        print(f"\nProcessing {patient}")

        # if patient == 'HCC_101' or patient == 'HCC_103':
        # 	continue

        for tp_idx, tp in enumerate(timepoint):
            tp_dir = os.path.join(IN_DIR, patient, tp)
            series = os.listdir(tp_dir)

            # Check if the directory contains segmentation
            if any([s.startswith("SEG") for s in series]):
                for series_idx, serie in enumerate(series):
                    # Get name of the series
                    serie_dir = os.path.join(tp_dir, serie)
                    first_img = os.listdir(serie_dir)[0]
                    dcm = dcmread(os.path.join(serie_dir, first_img))
                    name = dcm.SeriesDescription
                    if "Recon 3" in name:
                        seg_serie = series[[s.startswith("SEG") for s in series].index(True)]
                        valid_pairs = (serie_dir, os.path.join(tp_dir, seg_serie))
                        print(f"Found valid pair: {valid_pairs}")

                        # Convert to nifti using dicom2nifti
                        series_dir = valid_pairs[0]
                        seg_dir = valid_pairs[1]
                        series_nii = os.path.join(OUT_DIR, "imagesTr", f"{patient}_0000.nii.gz")
                        seg_nii = os.path.join(OUT_DIR, "labelsTr", f"{patient}.nii.gz")
                        print(f"Converting {series_dir} to {series_nii}")

                        # dicom2nifti.dicom_series_to_nifti(series_dir, series_nii,reorient_nifti=True)
                        # dcm_seg = dcmread(os.path.join(seg_dir, os.listdir(seg_dir)[0]))
                        # seg = dcm_seg.pixel_array
                        # seg = seg.astype(int)
                        # sitk.WriteImage(sitk.GetImageFromArray(seg), seg_nii)
                        # # Save as nifit

                        # reader = sitk.ImageFileReader()
                        # reader.SetFileName(os.path.join(seg_dir, os.listdir(seg_dir)[0]))
                        # image = reader.Execute()
                        # sitk.WriteImage(image, seg_nii)

                        # command = [". ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/MitkFileConverter.sh", "-i", str(os.path.join(seg_dir, os.listdir(seg_dir)[0])), "-o", str(seg_nii.replace('.nii.gz', '.nrrd'))]
                        # subprocess.call(command)

                        file_path = os.path.dirname(os.path.abspath(__file__))
                        in_img = os.path.join(file_path, series_dir, os.listdir(series_dir)[0])
                        out_img = os.path.join(file_path, series_nii)

                        # Convert image
                        os.system(
                            f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {in_img} -o {out_img}"
                        )

                        sitk_ct = sitk.ReadImage(out_img)
                        ct = sitk.GetArrayFromImage(sitk_ct)

                        in_seg = os.path.join(file_path, seg_dir, os.listdir(seg_dir)[0])
                        out_seg = os.path.join(file_path, seg_nii.replace(".nii.gz", ".nrrd"))

                        os.system(
                            f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {in_seg} -o {out_seg}"
                        )

                        # Convert nrrd to nifti
                        sitk_img = sitk.ReadImage(out_seg)
                        data = sitk.GetArrayFromImage(sitk_img)
                        data = data[:, :, :, 1] + data[:, :, :, 0] + data[:, :, :, 2]  # sum the three channels
                        data = np.where(data > 0, 1, 0)  # binarize the segmentation

                        if ct.shape != data.shape:
                            print(f"Shape mismatch for {patient} {tp}")
                            # delete the nifti files
                            os.remove(out_img)
                            continue

                        # connected component analysis

                        # data = cc3d.connected_components(data, connectivity=18)

                        out_img = sitk.GetImageFromArray(data)
                        out_img.SetSpacing(sitk_img.GetSpacing())
                        out_img.SetOrigin(sitk_img.GetOrigin())
                        out_img.SetDirection(sitk_img.GetDirection())
                        sitk.WriteImage(out_img, out_seg.replace(".nrrd", ".nii.gz"))
