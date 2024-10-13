
![INTRABENCH](assets/images/intrabench.png)
---
The IntRa Bench provides the medical segmentation field a comprehensive benchmark, that fairly compares existing open-set interactive segmentation methods on a broad-range of tasks and modalities.
The benchmark is intended to provide users with a clear understanding of how to best prompt existing methods and provides developers with a extendaable framework to easily compare their newly developed methods against the currently available methods.


## Installation


## Usage
To use the benchmark three steps need to be conducted:
### 1. Downloading the datasets
The datasets used in the benchmark can be downloaded using the following command:

```python
python ./src/intrab/datasets_preprocessing/download_all_datasets.py
# or only download a subset of datasets
python ./src/intrab/datasets_preprocessing/download_all_datasets.py --datasets ms_flair hanseg # can be multiple
```

Regarding selective downloads one can choose from:
  `["segrap", "hanseg", "ms_flair", "hntsmrg", "hcc_tace", "adrenal_acc", "rider_lung", "colorectal", "lnq", "pengwin"]`

### 2. Preprocessing the dataset
The dataset is often provided in a raw format, e.g. DICOMs which are not directly usable and can be a pain to deal with. To simplify things we provide preprocessing schemes that convert these directly to easier useable formats. The preprocessing can be done using the following commands.

```python
python ./src/intrab/datasets_preprocessing/preprocess_datasets.py --datasets ms_flair hanseg  # can be multiple
```

or again any choice of datasets from the list below:
`ms_flair, hanseg, hntsmrg, pengwin, segrap, lnq, colorectal, adrenal_acc, hcc_tace, rider_lung`

### 3. Running the benchmark
The benchmark for the `ms_flair` dataset and the `SAM` model can be run using the following command.

```python
python ./src/intrab/experiments_runner.py --config ./configs/static_prompt_SAMNORM_D1.yaml
```

Other configs can also be selected, but this can serve as an exemplary command to understand the benchmarking process.
