# Segment with 2d models and obtain results
source /home/t722s/anaconda3/etc/profile.d/conda.sh

results_dir="/media/t722s/2.0 TB Hard Disk/lesions_experiments_20240806/"

run_experiment_2d() {
    model=$1
    dataset_dir=$2
    python lesions_experiments_runner_2d.py -m $model -d $dataset_dir -r "$results_dir"
    python lesions_postprocess.py -m $model -d $dataset_dir -r "$results_dir"
}

run_experiment_segvol() {
    dataset_dir=$1
    python lesions_experiments_runner_segvol.py -m segvol -d $dataset_dir -r "$results_dir"
    python lesions_postprocess.py -m segvol -d $dataset_dir -r "$results_dir"
}


# SegVol experiments
conda activate segVol
run_experiment_segvol "/home/t722s/Desktop/Datasets/melanoma_HD/"
run_experiment_segvol "/home/t722s/Desktop/Datasets/Adrenal-ACC-Ki67/"
run_experiment_segvol "/home/t722s/Desktop/Datasets/Colorectal-Liver-Metastases/"

# 2d model experiments
conda activate foundationModels12-2

# Liver Lesions experiments
dataset_dir="/home/t722s/Desktop/Datasets/Colorectal-Liver-Metastases/"

# Execute for different models
run_experiment_2d "sam" $dataset_dir
run_experiment_2d "sammed2d" $dataset_dir
run_experiment_2d "medsam" $dataset_dir

# Melanoma experiments
dataset_dir="/home/t722s/Desktop/Datasets/melanoma_HD/"

# Execute for different models
run_experiment_2d "sam" $dataset_dir
run_experiment_2d "sammed2d" $dataset_dir
run_experiment_2d "medsam" $dataset_dir

# Adrenal ACC experiments
dataset_dir="/home/t722s/Desktop/Datasets/Adrenal-ACC-Ki67/"

# Execute for different models
run_experiment_2d "sam" $dataset_dir
run_experiment_2d "sammed2d" $dataset_dir
run_experiment_2d "medsam" $dataset_dir
