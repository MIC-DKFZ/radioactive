from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger
import nrrd
import numpy as np
from tqdm import tqdm

from intrab.datasets_preprocessing.conversion_utils import (
    dicom_to_nrrd,
    get_dicoms_meta_info,
    get_matching_img,
    read_dicom_meta_data,
)
from intrab.utils.paths import get_dataset_path
from toinstance import InstanceNrrd
import SimpleITK as sitk

included_cases = [
    "HCC_004",
    "HCC_012",
    "HCC_013",
    "HCC_017",
    "HCC_020",
    "HCC_022",
    "HCC_023",
    "HCC_025",
    "HCC_026",
    "HCC_028",
    "HCC_029",
    "HCC_031",
    "HCC_032",
    "HCC_033",
    "HCC_034",
    "HCC_035",
    "HCC_036",
    "HCC_037",
    "HCC_038",
    "HCC_040",
    "HCC_041",
    "HCC_042",
    "HCC_043",
    "HCC_044",
    "HCC_045",
    "HCC_046",
    "HCC_047",
    "HCC_048",
    "HCC_049",
    "HCC_050",
    "HCC_051",
    "HCC_052",
    "HCC_053",
    "HCC_055",
    "HCC_056",
    "HCC_057",
    "HCC_058",
    "HCC_059",
    "HCC_060",
    "HCC_061",
    "HCC_062",
    "HCC_063",
    "HCC_064",
    "HCC_066",
    "HCC_067",
    "HCC_069",
    "HCC_070",
    "HCC_071",
    "HCC_072",
    "HCC_073",
    "HCC_074",
    "HCC_075",
    "HCC_076",
    "HCC_077",
    "HCC_078",
    "HCC_079",
    "HCC_080",
    "HCC_081",
    "HCC_082",
    "HCC_083",
    "HCC_084",
    "HCC_086",
    "HCC_087",
    "HCC_088",
    "HCC_090",
    "HCC_098",
]


def preprocess(raw_download_dir: Path):

    IN_DIR = raw_download_dir / "hcc_tace_seg"

    liver_output_dir = get_dataset_path() / "Dataset920_hcc_tace_liver"
    lesion_output_dir = get_dataset_path() / "Dataset921_hcc_tace_lesion"

    patients = os.listdir(IN_DIR)

    for patient in sorted(patients):
        timepoint = IN_DIR / patient
        print(f"\nProcessing {patient}")

        # if patient not in included_cases:
        #     continue

        for tp_idx, tp in enumerate(list(timepoint.iterdir())):
            tp_dir: Path = IN_DIR / patient / tp  # os.path.join(IN_DIR, patient, tp)
            series = os.listdir(str(tp_dir))

            # Check if the directory contains segmentation
            if any([s.startswith("SEG") for s in series]):
                ct_headers = {
                    serie: read_dicom_meta_data(tp_dir / serie) for serie in series if serie.startswith("CT")
                }
                ct_of_interests = {
                    serie: ct_h for serie, ct_h in ct_headers.items() if "Recon 3" in ct_h.SeriesDescription
                }
                if len(ct_of_interests) != 1:
                    # logger.info(f"Found {len(ct_of_interests)} 'Recon 3' CTs - skipping")
                    continue

                ct_path = list(ct_of_interests.keys())[0]
                ct_of_interest = list(ct_of_interests.values())[0].SeriesInstanceUID
                seg_headers = {
                    serie: read_dicom_meta_data(tp_dir / serie) for serie in series if serie.startswith("SEG")
                }
                seg_of_interests = {
                    serie: seg_h
                    for serie, seg_h in seg_headers.items()
                    if seg_h.ReferencedSeriesSequence[0].SeriesInstanceUID == ct_of_interest
                }

                seg_of_interests = {
                    serie: seg_h
                    for serie, seg_h in seg_of_interests.items()
                    if seg_h.SeriesDescription == "Segmentation"
                }
                if len(seg_of_interests) != 1:
                    # logger.info(f"Found {len(seg_of_interests)} SEGs - skipping")
                    continue
                seg_path = list(seg_of_interests.keys())[0]

                ct = dicom_to_nrrd(tp_dir / ct_path)
                orig_seg = dicom_to_nrrd(tp_dir / seg_path)
                (liver_output_dir / "orig").mkdir(parents=True, exist_ok=True)
                # (liver_output_dir / "orig_seg").mkdir(parents=True, exist_ok=True)
                for tmp_ct_path, tmp_ct_h in ct_headers.items():
                    tmp_ct = dicom_to_nrrd(tp_dir / tmp_ct_path)
                    nrrd.write(
                        str(
                            liver_output_dir
                            / "orig"
                            / f"ct_{tmp_ct_h.SeriesDescription.replace("/","")}_{patient}.nrrd"
                        ),
                        tmp_ct[0],
                        tmp_ct[1],
                    )
                # nrrd.write(str(liver_output_dir / "orig_ct" / f"{patient}.nrrd"), ct[0], ct[1])
                nrrd.write(str(liver_output_dir / "orig" / f"{patient}.nrrd"), orig_seg[0], orig_seg[1])

                if ct[0].shape != orig_seg[0].shape[1:]:
                    print(f"Shape mismatch for {patient} {tp}")
                    # delete the nifti files
                    continue

                keys_in_labels = {
                    group["labels"][0]["name"]: cnt
                    for cnt, group in enumerate(
                        json.loads(orig_seg[1]["org.mitk.multilabel.segmentation.labelgroups"])
                    )
                }

                print(f"Processing patient {patient} at timepoint {tp}")
                for output_dir, sem_class in [(liver_output_dir, "liver"), (lesion_output_dir, "lestion")]:
                    images_dir = output_dir / "imagesTr"
                    labels_dir = output_dir / "labelsTr"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    labels_dir.mkdir(parents=True, exist_ok=True)

                    # Seg holds: group 1: Liver, group 2: Lesion, group3: hepatic vessels, group4: abdominal aorta
                    # We copy over the CT header to remove original 4D MITK header stuff
                    if sem_class == "liver":
                        keys_of_interest = [
                            v for k, v in keys_in_labels.items() if k in ["Necrosis", "Mass", "Portal vein", "Liver"]
                        ]
                        seg = (
                            np.where(np.sum(orig_seg[0][keys_of_interest], axis=0) != 0, 1, 0),
                            ct[1],
                        )
                        innrrd = InstanceNrrd.from_semantic_map(
                            semantic_map=seg[0],
                            header=deepcopy(seg[1]),
                            do_cc=False,
                        )
                    elif sem_class == "lesion":
                        keys_of_interest = [v for k, v in keys_in_labels.items() if k in ["Mass", "Necrosis"]]
                        seg = (np.where(orig_seg[0][keys_of_interest] != 0, 1, 0), ct[1])
                        innrrd = InstanceNrrd.from_semantic_map(
                            semantic_map=seg[0],
                            header=deepcopy(seg[1]),
                            do_cc=False,
                            cc_kwargs={"dilation_kernel_radius": 0, "label_connectivity": 3},
                        )

                    innrrd.to_file(labels_dir / f"{patient}.nrrd")
                    nrrd.write(str(images_dir / f"{patient}_0000.nrrd"), ct[0], ct[1])

    # print(f"CT: {ct_of_interest}\n SEG:{seg_of_interest}\n\n")
    # for series_idx, serie in enumerate([s for s in series if s.startswith("SEG")]):
    #     # Get name of the series
    #     serie_dir = tp_dir / serie  # os.path.join(tp_dir, serie)
    #     first_img = os.listdir(str(serie_dir))[0]
    #     dcm = read_dicom_meta_data(serie_dir / first_img)

    #     headers = [read_dicom_meta_data(serie_dir / img) for img in os.listdir(str(serie_dir))]
    #     name = dcm.SeriesDescription
    #     if "Recon 3" in name:
    #         seg_serie = series[[s.startswith("SEG") for s in series].index(True)]
    #         valid_pairs = (serie_dir, tp_dir / seg_serie)
    #         print(f"Found valid pair: {valid_pairs}")

    #         # Convert to nifti using dicom2nifti
    #         series_dir = valid_pairs[0]
    #         seg_dir = valid_pairs[1]
    #         ct = dicom_to_nrrd(series_dir)
    #         seg = dicom_to_nrrd(seg_dir)

    #         if ct[0].shape != seg[0].shape:
    #             print(f"Shape mismatch for {patient} {tp}")
    #             # delete the nifti files
    #             continue

    #         for output_dir, sem_class in [(liver_output_dir, "liver"), (lesion_output_dir, "lestion")]:
    #             images_dir = output_dir / "imagesTr"
    #             labels_dir = output_dir / "labelsTr"
    #             images_dir.mkdir(parents=True, exist_ok=True)
    #             labels_dir.mkdir(parents=True, exist_ok=True)
    #             if sem_class == "liver":
    #                 # data = data[:,:,:,1] + data[:,:,:,0] + data[:,:,:,2] # sum the three channels
    #                 # Join all the channels
    #                 seg = (np.where(np.sum(seg[0], axis=0) > 1, 1, 0), seg[1])
    #             elif sem_class == "lesion":
    #                 pass
    #             # ------------------------------- Write the CT ------------------------------- #
    #             nrrd.write(str(images_dir / f"{patient}_0000.nrrd"), ct[0], ct[1])
    #             # ------------------- Create instanced version and save Seg ------------------ #
    #             with TemporaryDirectory() as tempdir:
    #                 innrrd = InstanceNrrd.from_semantic_map(
    #                     semantic_map=seg[0],
    #                     header=deepcopy(seg[1]),
    #                     do_cc=True,
    #                     cc_kwargs={"dilation_kernel_radius": 0, "label_connectivity": 3},
    #                 )
    #                 innrrd.to_file(str(labels_dir / f"{patient}.nrrd"))
    #                 # seg = sitk.ReadImage(str(Path(tempdir) / "tmp.nrrd"))
    #                 # nrrd.write(str(Path(tempdir) / "tmp_ct.nrrd"), ct[0], ct[1])
    #                 # temp_ct = sitk.ReadImage(str(Path(tempdir) / "tmp_ct.nrrd"))
    #                 # padded_seg = resample_to_match(reference_img=temp_ct, resample_img=seg, is_seg=True)
    #                 # sitk.WriteImage(seg, str(labels_dir / f"{patient}.nrrd"))

    #             # connected component analysis

    #             # data = cc3d.connected_components(data, connectivity=18)

    #             out_img = sitk.GetImageFromArray(data)
    #             out_img.SetSpacing(sitk_img.GetSpacing())
    #             out_img.SetOrigin(sitk_img.GetOrigin())
    #             out_img.SetDirection(sitk_img.GetDirection())
    #             sitk.WriteImage(out_img, out_seg.replace(".nrrd", ".nii.gz"))

    # patient_dirs = [p for p in raw_root_dir.iterdir() if (p.is_dir() and p.name.startswith("HCC"))]

    # for patient_dir in tqdm(patient_dirs, desc="Converting LNQ DICOMs to NRRD"):
    #     study_name= patient_dir.name
    #     if study_name not in included_cases:
    #         continue
    #     # We only have one SEG and an associated CT
    #     meta_info = get_dicoms_meta_info(list(patient_dir.iterdir()))
    #     # Only one CT and one SEG in the dir
    #     ct_path = meta_info["CT"][0]["filepath"]
    #     seg_path = meta_info["SEG"][0]["filepath"]

    #     ct: tuple[np.ndarray, dict] = dicom_to_nrrd(ct_path)
    #     seg: tuple[np.ndarray, dict] = dicom_to_nrrd(seg_path)

    #     for output_dir, sem_class in [(liver_output_dir, "liver"), (lesion_output_dir, "lestion")]:
    #         images_dir = output_dir / "imagesTr"
    #         labels_dir = output_dir / "labelsTr"
    #         images_dir.mkdir(parents=True, exist_ok=True)
    #         labels_dir.mkdir(parents=True, exist_ok=True)
    #         if sem_class == "liver":
    #             # data = data[:,:,:,1] + data[:,:,:,0] + data[:,:,:,2] # sum the three channels
    #             # Join all the channels
    #             seg = (np.where(np.sum(seg[0], axis=0)>1, 1, 0), seg[1])
    #         elif sem_class == "lesion":
    #             pass

    #         innrrd = InstanceNrrd.from_semantic_map(
    #             semantic_map=seg[0],
    #             header=deepcopy(seg[1]),
    #             do_cc=True,
    #             cc_kwargs={"dilation_kernel_radius": 0, "label_connectivity": 3},
    #         )
    #         innrrd.to_file(labels_dir / f"HCC_{cnt:03d}.nrrd")
    #         nrrd.write(str(images_dir / f"HCC_{cnt:03d}_0000.nrrd"), ct[0], ct[1])

    # # ------------------------------- Dataset Json ------------------------------- #
    # for output_dir in [(liver_output_dir), (lesion_output_dir)]:
    #     with open(output_dir / "dataset.json", "w") as f:
    #         json.dump(
    #             {
    #                 "channel_names": {"0": "CT"},
    #                 "labels": {"background": 0, f"{output_dir.name.split("_")}": 1},
    #                 "numTraining": len(list(all_dicoms)),
    #                 "file_ending": ".nrrd",
    #                 "name": f"{output_dir.name}",
    #             },
    #             f,
    #         )
    #     with open(output_dir / "study_name_patient_id_map.json", "w") as f:
    #         json.dump(cnt_dicom_map, f)

    # Get Image Label Pairs
