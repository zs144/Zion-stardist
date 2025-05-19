python3 ../zion_scripts/infer_main_2d.py \
    --model_ckpt_path "/hpc/group/yizhanglab/zs144/repo/Zion-stardist/models/python_2D_versatile_he/" \
    --image_name "STHD_bagel_example" \
    --test_image_filepath "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/projects/semi_seg/image/h&e_crop10.png" \
    --prob_threshold 0.03 \
    --scale 1.5 \
    --output_dir "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/projects/semi_seg/output/stardist/stardist_v0_run7" \

echo "Done"
