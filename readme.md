# Universal Models
This repository will contain a universal interface for a few segmentation models (listed below) to assess and compare their performances.

## Models
### SAM-Adapted

[SAM](https://github.com/facebookresearch/segment-anything) (WIP, branch: sam)

[MedSAM](needlink) (To do)

[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) (To do)

[SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) (To do)

### Other

[Universeg](https://github.com/JJGO/UniverSeg) (To do)

[SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) (To do)

Checkpoint from SAM-Med3D obtained on the relevant repository

## Directory Structure
List requirements on directory strucutre (ie Ts, nifti files, and in line with AMOS that x is the superior axis. Hopefully can change that)

## Use
Point prompts are generated first to permit using the same prompts for each model. Generate them with `generate_points.py` taking flags
- -tdp, or --test_data_path: Where the query images are stored (in the directory structure specified in Directory Strucutre)
- -rp, or --results_path: Where the prompts file is to be stored. Use the same directory in `validation.py` for the segmentation maps
- -nc, or --n_clicks [default = 5]: The number of clicks to generate per volume per foreground label (for 3D models) or per slice containing foreground per foreground label (for 2D models). (to implement: allow the first click to be used as a seed prompt and use cleverer interactive point generation methods than just the random generation.)

For example:
```
python generate_points.py \
    -tdp /home/t722s/Desktop/Datasets/BratsTestData/ \
    -rp /home/t722s/Desktop/Sam-Med3DTest/evalBrats/ \
    -nc 5
```