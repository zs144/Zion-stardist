SLIDE_ID="240819_Ji_N1_H_EScan"

python3 ../zion_scripts/infer_main_2d.py \
    --model_ckpt_path "/hpc/group/yizhanglab/zs144/repo/Zion-stardist/examples/2D/models/models_v0.9/" \
    --image_name $SLIDE_ID \
    --wsi_filepath "/hpc/group/yizhanglab/DATA/zs144/MedSAM/neuron_wsi/$SLIDE_ID.tif" \
    --patch_len 2048 \
    --target_len 256 \
    --prob_threshold 0.2 \
    --output_dir "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/projects/segment_neurons/output/$SLIDE_ID/pred_whole_mask/" \

echo "Done"
