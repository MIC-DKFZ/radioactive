# Segment with 2d models and obtain results
results_dir="/media/t722s/2.0 TB Hard Disk/lesions_experiments_20240730/"
run_experiment() {
    model=$1
    dataset_dir=$2
    python lesions_experiments_runner_2d.py -m $model -d $dataset_dir -r "$results_dir"
    python lesions_postprocess.py -m $model -d $dataset_dir -r "$results_dir"
}

# Melanoma experiments
dataset_dir="/home/t722s/Desktop/Datasets/melanoma_HD/"

# Execute for different models
run_experiment "sam" $dataset_dir
run_experiment "sammed2d" $dataset_dir
run_experiment "medsam" $dataset_dir

# # Adrenal ACC experiments
# dataset_dir="/home/t722s/Desktop/Datasets/Adrenal-ACC-Ki67/"

# # Execute for different models
# run_experiment "sam" $dataset_dir
# run_experiment "sammed2d" $dataset_dir
# run_experiment "medsam" $dataset_dir
