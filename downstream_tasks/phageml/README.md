# PhageML

## Training

1. Go to the ModernBERT repository root:
   ```bash
   cd /path/to/ModernBERT
   ```

2. Configure the paths in your .sh script:
WORK_DIR — must be the path to the ModernBERT repository
OUTPUT_DIR — must be the path to the main directory for saving results

3. Run the training script:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 ./downstream_tasks/phageml/regressor_multigpu/run_finetuning_bpe_1.sh
   ```
   