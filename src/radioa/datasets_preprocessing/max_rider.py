import os
import SimpleITK as sitk
import pydicom

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    seg_dir = "seg"
    img_dir = "img"

    img_nifti_dir = "img_nifti"
    seg_nrrd_dir = "seg_nrrd"
    seg_nifti_dir = "seg_nifti"
    temp_nrrd_dir = "temp_nrrd"

    os.makedirs(img_nifti_dir, exist_ok=True)
    os.makedirs(seg_nrrd_dir, exist_ok=True)
    os.makedirs(seg_nifti_dir, exist_ok=True)
    os.makedirs(temp_nrrd_dir, exist_ok=True)

    img_nifti_dir = os.path.join(script_dir, img_nifti_dir)
    seg_nrrd_dir = os.path.join(script_dir, seg_nrrd_dir)
    seg_nifti_dir = os.path.join(script_dir, seg_nifti_dir)
    temp_nrrd_dir = os.path.join(script_dir, temp_nrrd_dir)

    for rider_pat in os.listdir(img_dir):
        scans = os.listdir(os.path.join(img_dir, rider_pat))
        if len(scans) == 1:
            scans_new = os.listdir(os.path.join(img_dir, rider_pat, scans[0]))
            if len(scans_new) == 1:
                print(rider_pat)
                print(scans_new)
                seg_path = os.path.join(seg_dir, rider_pat, scans[0])
                candidates = [f for f in os.listdir(seg_path) if "TEST" in f and "RIDER" in f]
                assert len(candidates) == 1

                # Convert img to nifti
                dicom__img_path = os.path.join(img_dir, rider_pat, scans[0], scans_new[0])
                dicom__img_path = os.path.join(script_dir, dicom__img_path, os.listdir(dicom__img_path)[0])
                dicom__seg_path = os.path.join(seg_dir, rider_pat, scans[0], candidates[0])
                dicom__seg_path = os.path.join(script_dir, dicom__seg_path, os.listdir(dicom__seg_path)[0])
                os.system(
                    f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom__img_path} -o {img_nifti_dir}/{rider_pat}_{scans[0]}.nii.gz"
                )
                os.system(
                    f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom__seg_path} -o {seg_nrrd_dir}/{rider_pat}_{scans[0]}.nrrd"
                )
            else:
                seg_path = os.path.join(seg_dir, rider_pat, scans[0])

                # find TEST/RETEST
                scan1_img, scan2_img = [os.path.join(img_dir, rider_pat, scans[0], s) for s in scans_new]
                print(scans_new)
                print(scan1_img)
                print(scan2_img)

                scan_1_z = len(os.listdir(scan1_img))
                scan_2_z = len(os.listdir(scan2_img))
                print(scan_1_z, scan_2_z)

                scan1_seg, scan2_seg = [
                    os.path.join(seg_path, s)
                    for s in [f for f in os.listdir(seg_path) if "TEST" in f and "RIDER" in f]
                ]

                dicom_file_path_scan1 = os.path.join(scan1_seg, os.listdir(scan1_seg)[0])
                dicom_file_path_scan2 = os.path.join(scan2_seg, os.listdir(scan2_seg)[0])

                # convert to nrrd
                dicom_file_path_scan1 = os.path.join(script_dir, dicom_file_path_scan1)
                dicom_file_path_scan2 = os.path.join(script_dir, dicom_file_path_scan2)
                os.system(
                    f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom_file_path_scan1} -o {temp_nrrd_dir}/{rider_pat}_{scans_new[0]}.nrrd"
                )
                os.system(
                    f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom_file_path_scan2} -o {temp_nrrd_dir}/{rider_pat}_{scans_new[1]}.nrrd"
                )

                scan1_seg_nrrd = os.path.join(temp_nrrd_dir, f"{rider_pat}_{scans_new[0]}.nrrd")
                scan2_seg_nrrd = os.path.join(temp_nrrd_dir, f"{rider_pat}_{scans_new[1]}.nrrd")

                # load nrrd
                scan1_seg_nrrd = sitk.ReadImage(scan1_seg_nrrd)
                scan2_seg_nrrd = sitk.ReadImage(scan2_seg_nrrd)

                if scan1_seg_nrrd.GetSize() == scan2_seg_nrrd.GetSize():
                    print("Equal")
                else:
                    if scan1_seg_nrrd.GetSize()[2] == scan_1_z and scan2_seg_nrrd.GetSize()[2] == scan_2_z:
                        # convert scan 1 img and scan 1 seg
                        dicom_file_path_scan1 = os.path.join(scan1_img, os.listdir(scan1_img)[0])
                        dicom_file_path_scan1 = os.path.join(script_dir, dicom_file_path_scan1)
                        os.system(
                            f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom_file_path_scan1} -o {img_nifti_dir}/{rider_pat}_{scans_new[0]}.nii.gz"
                        )
                        sitk.WriteImage(
                            scan1_seg_nrrd, os.path.join(seg_nrrd_dir, f"{rider_pat}_{scans_new[0]}.nrrd")
                        )

                        # convert scan 2 img and scan 2 seg
                        dicom_file_path_scan2 = os.path.join(scan2_img, os.listdir(scan2_img)[0])
                        dicom_file_path_scan2 = os.path.join(script_dir, dicom_file_path_scan2)
                        os.system(
                            f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom_file_path_scan2} -o {img_nifti_dir}/{rider_pat}_{scans_new[1]}.nii.gz"
                        )
                        sitk.WriteImage(
                            scan2_seg_nrrd, os.path.join(seg_nrrd_dir, f"{rider_pat}_{scans_new[1]}.nrrd")
                        )

                    elif scan1_seg_nrrd.GetSize()[2] == scan_2_z and scan2_seg_nrrd.GetSize()[2] == scan_1_z:
                        # Other way around

                        # convert scan 1 img and scan 2 seg
                        dicom_file_path_scan1 = os.path.join(scan1_img, os.listdir(scan1_img)[0])
                        dicom_file_path_scan1 = os.path.join(script_dir, dicom_file_path_scan1)
                        os.system(
                            f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom_file_path_scan1} -o {img_nifti_dir}/{rider_pat}_{scans_new[0]}.nii.gz"
                        )
                        sitk.WriteImage(
                            scan2_seg_nrrd, os.path.join(seg_nrrd_dir, f"{rider_pat}_{scans_new[0]}.nrrd")
                        )

                        # convert scan 2 img and scan 1 seg
                        dicom_file_path_scan2 = os.path.join(scan2_img, os.listdir(scan2_img)[0])
                        dicom_file_path_scan2 = os.path.join(script_dir, dicom_file_path_scan2)
                        os.system(
                            f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom_file_path_scan2} -o {img_nifti_dir}/{rider_pat}_{scans_new[1]}.nii.gz"
                        )
                        sitk.WriteImage(
                            scan1_seg_nrrd, os.path.join(seg_nrrd_dir, f"{rider_pat}_{scans_new[1]}.nrrd")
                        )

                    else:
                        raise ValueError("Z mismatch")

        else:
            for scan in scans:
                print(rider_pat)
                img_path = os.path.join(img_dir, rider_pat, scan)
                assert len(os.listdir(img_path)) == 1
                print(os.listdir(img_path))
                seg_path = os.path.join(seg_dir, rider_pat, scan)
                candidates = [f for f in os.listdir(seg_path) if "TEST" in f and "RIDER" in f]
                assert len(candidates) == 1

                # Convert img to nifti
                dicom__img_path = os.path.join(img_dir, rider_pat, scan, os.listdir(img_path)[0])
                dicom__img_path = os.path.join(script_dir, dicom__img_path, os.listdir(dicom__img_path)[0])
                dicom__seg_path = os.path.join(seg_dir, rider_pat, scan, candidates[0])
                dicom__seg_path = os.path.join(script_dir, dicom__seg_path, os.listdir(dicom__seg_path)[0])
                os.system(
                    f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom__img_path} -o {img_nifti_dir}/{rider_pat}_{scan}.nii.gz"
                )
                os.system(
                    f"cd ~/Downloads/MITK-snapshots_2024-08-08-linux-x86_64/apps/ && ./MitkFileConverter.sh -i {dicom__seg_path} -o {seg_nrrd_dir}/{rider_pat}_{scan}.nrrd"
                )

    #### not tested together below this line, should work together though

    out_lbl_dir = "labelsTr"
    os.makedirs(out_lbl_dir, exist_ok=True)
    import cc3d
    import numpy as np

    for lbl in os.listdir(seg_nrrd_dir):
        if os.path.exists(os.path.join(out_lbl_dir, lbl.replace(".nrrd", ".nii.gz"))):
            continue
        lbl_path = os.path.join(seg_nrrd_dir, lbl)
        lbl_nrrd = sitk.ReadImage(lbl_path)
        lbl_nrrd_data = sitk.GetArrayFromImage(lbl_nrrd)

        try:
            if len(lbl_nrrd_data.shape) == 4:
                lbl_nrrd_data = lbl_nrrd_data[:, :, :, 1]
            elif len(lbl_nrrd_data.shape) == 3:
                lbl_nrrd_data = lbl_nrrd_data
        except Exception as e:
            print(lbl)
            raise e

        # binarize
        lbl_nrrd_data[lbl_nrrd_data > 0] = 1
        lbl_nrrd_data = lbl_nrrd_data.astype(np.int8)
        lbl_nrrd_data_cc, num = cc3d.connected_components(lbl_nrrd_data, return_N=True)
        if num != 1:
            print(f"Number of connected components: {num} in {lbl}")

        # save as nifti
        lbl_out = sitk.GetImageFromArray(lbl_nrrd_data)
        lbl_out.SetSpacing(lbl_nrrd.GetSpacing())
        lbl_out.SetOrigin(lbl_nrrd.GetOrigin())
        lbl_out.SetDirection(lbl_nrrd.GetDirection())
        sitk.WriteImage(lbl_out, os.path.join(out_lbl_dir, lbl.replace(".nrrd", ".nii.gz")))
