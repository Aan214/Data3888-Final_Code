# Cell Image Classification Experiment

This project implements a comparative study of multiple deep learning models (CNN, ViT, Swin Transformer, MLP) for cell image classification tasks.

## Requirements

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Raw Data Structure:
```
dataset_raw/
├── metadata_code/
│   └── cbr.csv           # Contains cell bounding box information
└── 100/                  # Contains raw cell images
    ├── B_Cells/
    ├── CD4+_T_Cells/
    └── ...
```

You need to put this at the root of this folder.

2. Data Processing Steps:

a. Split the data into train/validation/test sets:
```bash
python split.py
```
This will:
- Read cell bounding box information from cbr.csv
- Group cells based on spatial overlap
- Split data into train, validation, and test sets
- Create the following structure:
```
dataset_splits/
├── train/
├── validation/
└── test/
```

b. Generate masks for the images (optional):
```bash
python gen_mask.py --src dataset_splits --dst dataset_splits_masked --ext png --gpu
```
This will:
- Use Cellpose to generate masks for each cell
- Save masked images in the same directory structure

## Running Steps

1. Data Preprocessing
```bash
python preprocessing.py
```
This will:
- Process the split datasets
- Generate train.txt, validation.txt, and test.txt files
- Balance the number of samples across categories

2. Run the Experiment
```bash
python experiment.py
```
This will:
- Train multiple models (CNN, ViT, Swin Transformer, MLP)
- Save the best models to the `saved_model` directory
- Generate performance comparison plots

3. Post-hoc Analysis (Optional)
```bash
python post_hoc.py --mode [gradcam|legrad|attn|shap|lime] --model_path saved_model/best_swin.pth --image_path path/to/image.png --num_classes 5 --model_name swin_base_patch4_window7_224 --output_dir results
```
This will:
- Generate visualization of model attention
- Save results in the specified output directory

## Output Results

- Model weights are saved in the `saved_model` directory:
  - `best_cnn.pth`
  - `best_vit.pth`
  - `best_swin.pth`
  - `best_mlp_mixer.pth`
- Performance comparison plots:
  - `loss_compare.png`: Training loss comparison
  - `results/accuracy_compare1.png`: Validation accuracy comparison
- Post-hoc analysis results in the `results` directory

## Notes

1. Ensure sufficient GPU memory (if using GPU)
2. You can modify the `EXCLUDE_CLUSTERS` list in `preprocessing.py` to exclude specific categories
3. You can comment/uncomment different model training sections in `experiment.py`
4. For mask generation, ensure you have enough GPU memory and adjust the batch size accordingly

## Troubleshooting

If you encounter issues:
1. Check if the data paths are correct
2. Verify all dependencies are installed
3. If GPU memory is insufficient:
   - Reduce the `batch_size` in experiment.py
   - Reduce the batch size in gen_mask.py
   - Use CPU instead of GPU for mask generation
4. If CPU processing is slow, adjust the `num_workers` parameter

## Project Structure

```
.
├── experiment.py          # Main experiment script
├── preprocessing.py       # Data preprocessing script
├── split.py              # Data splitting script
├── gen_mask.py           # Mask generation script
├── post_hoc.py           # Post-hoc analysis script
├── train/                # Training related code
│   ├── cnn_training.py
│   ├── vit_training.py
│   ├── swin_training.py
│   └── mlp_training.py
├── dataset_raw/          # Raw dataset directory
├── dataset_splits/       # Processed dataset directory
└── saved_model/         # Model save directory
``` 