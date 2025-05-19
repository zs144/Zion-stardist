VERSION="v1.0"

python3 ../zion_scripts/train_main_2d_v2.py \
    --exp_id "test_run_$VERSION" \
    --wandb_run_name "stardist-h&e2d-$VERSION" \
    --data_path "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/projects/segment_neurons/output/" \
    --slides "240819_Ji_D2N3_H_EScan" "240819_Ji_N1_H_EScan" "240819_Ji_N2D3_H_EScan" \
    --ckpt_path "/hpc/group/yizhanglab/zs144/repo/Zion-stardist/zion_assets/models/test_run_$VERSION/" \
    --use_gpu \
    --use_wandb \
    --train_batch_size 4 \
    --n_epochs 100 \
