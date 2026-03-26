# PhageML

## Training

1. Go to the ModernBERT repository root:
   ```bash
   cd /path/to/ModernBERT
   ```
2. Download the conda environment archive:
    ```bash
    s3://genalm/phageml/bert24_artem.tar.gz
    ```
3. Download checkpoint
    ```bash
    aws s3 cp s3://genalm/phageml/ckpts/base_metavr_bpe_continue/ \
        ModernBERT/downstream_tasks/phageml/ckpt/base_metavr_bpe_continue/ \
        --endpoint-url https://s3.cloud.ru \
        --profile airi \
        --recursive
    ```
4. Configure the paths in your .sh script:
WORK_DIR — must be the path to the ModernBERT repository
OUTPUT_DIR — must be the path to the main directory for saving results

5. Run the training script:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 ./downstream_tasks/phageml/regressor_multigpu/run_finetuning_bpe_1.sh
   ```
   