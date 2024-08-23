# **IntRa** Bench: Introducing the **Int**eractive **Ra**diology Benchmark
The IntRa Bench provides the medical segmentation field a comprehensive benchmark, that fairly compares existing open-set interactive segmentatio methods on a broad-range of tasks and modalities. 
The benchmark is intended to provide users with a clear understanding of how to best prompt existing methods and provides developers with a extendaable framework to easily compare their newly developed methods against the currently available methods.

## Installing the Benchmark
ToDo

## Extending the Benchmark
ToDo

## Currently supported models
ToDo

[SAM](https://github.com/facebookresearch/segment-anything) (WIP, branch: sam)

[MedSAM](needlink) (To do)

[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) (To do)

[SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) (To do)

### Other

[Universeg](https://github.com/JJGO/UniverSeg) (To do)

[SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) (To do)

Checkpoint from SAM-Med3D obtained on the relevant repository

## Directory Structure
Please store your volumes and labels like
data
  ├── imagesTs
  │ ├── word_0025.nii.gz
  │ ├── ...
  ├── labelsTs
  │ ├── word_0025.nii.gz
  │ ├── ...
  ├── ...
  ├── dataset.json

In particular:
* Images are only drawn from the TEST folders, so imagesTr and LabelsTr will be ignored
* Only nifti files are supported currently
* dataset.json file must be included, and is used to find the foreground labels and what they correspond to.

## Use
### Prompt Generation

Point prompts are generated first to permit using the same prompts for each model. Generate them with `generate_points.py` taking flags
- -tdp, or --test_data_path: Where the query images are stored (in the directory structure specified in Directory Strucutre)
- -rp, or --results_path: Where the prompts file should be stored. Use the same directory in `validation.py` for the segmentation maps.
- -nc, or --n_clicks [default = 5]: The number of clicks to generate per volume per foreground label (for 3D models) or per slice containing foreground per foreground label (for 2D models). 

For example:
```
python generate_points.py \
    -tdp /home/t722s/Desktop/Datasets/BratsMini/ \
    -rp /home/t722s/Desktop/Sam-Med3DTest/evalBrats/ \
    -nc 5
```

### Inference
Perform inference with `validation_interface.py` (in progress. Use validation_interface.ipynb for now)

### Evaluation
Obtain evaluation results with `evaluate_folder.py`. Requires flags
- -tdp, or --test_data_path: Where the test data are stored (format as in Directory Structure)
- -rp, or --results_path: Where the segmentations are stored. Use the same directory in `validation.py` 

Evaluation results will be stored in the `-rp` folder as `evaluation_dice.json`

```
python evaluate_folder.py \
    -tdp /home/t722s/Desktop/Datasets/BratsMini/ \
    -rp /home/t722s/Desktop/Sam-Med3DTest/evalBrats
```
