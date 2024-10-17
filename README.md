
![INTRABENCH](assets/images/intrabench.png)
---
The **Int**eractive **Ra**diology **Bench**chmark allows open-set interactive 2D or 3D segmentation methods to evaluate themselves fairly against other methods on the field of radiological images. **IntRaBench** currently includes _6 interactive segmentation methods_, spans _eight datasets_ (including CT and MRI) with _various anatomical and pathological targets_.

Through this benchmark, we provide users with transparent results on what the best existing methods are and provide developers an extendable framework, allowing them to easily compare their newly developed models or prompting schemes against currently available methods.

## Installation
1. Activate virtualenv of choice (with e.g. python 3.12)
2. Clone & install SAM2 `git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .`
3. Download IntRaBench repository (clone or download and extract manually)
4. `cd intrabench && pip install -e .`
5. Done.

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
