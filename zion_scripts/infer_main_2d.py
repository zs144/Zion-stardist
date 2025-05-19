import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage.transform import resize
from csbdeep.utils import normalize_mi_ma
import openslide
from openslide.deepzoom import DeepZoomGenerator
from stardist import random_label_cmap
from stardist.models import Config2D, StarDist2D
from time import time

np.random.seed(42)

parser = argparse.ArgumentParser(description='Inference with StarDist')
parser.add_argument("--model_ckpt_path", type=str, required=True)
parser.add_argument("--image_name", type=str, required=True)
parser.add_argument("--wsi_filepath", type=str)
parser.add_argument("--test_image_filepath", type=str)
parser.add_argument("--patch_len", type=int, default=2048)
parser.add_argument("--target_len", type=int, default=256,
                    help="Target image/patch size for the model")
parser.add_argument("--prob_threshold", type=float, default=0.5,
                    help="Probability threshold for the neuron cell prediction")
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--output_dir", type=str, required=True)

args = parser.parse_args()
model_ckpt_path     = args.model_ckpt_path
image_name          = args.image_name
wsi_filepath        = args.wsi_filepath
test_image_filepath = args.test_image_filepath
patch_len           = args.patch_len
target_len          = args.target_len
prob_threshold      = args.prob_threshold
scale               = args.scale
output_dir          = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save settings to a txt file
settings_file = os.path.join(output_dir, 'settings.txt')
with open(settings_file, 'w') as f:
    f.write(f"model_ckpt_path: {model_ckpt_path}\n")
    f.write(f"image_name: {image_name}\n")
    if wsi_filepath:
        f.write(f"wsi_filepath: {wsi_filepath}\n")
        f.write(f"patch_len: {patch_len}\n")
        f.write(f"target_len: {target_len}\n")
    if test_image_filepath:
        f.write(f"test_image_filepath: {test_image_filepath}\n")
    f.write(f"prob_threshold: {prob_threshold}\n")
    f.write(f"scale: {scale}\n")

start = time()

# Load pretrained model
model = StarDist2D(None, name='stardist', basedir=model_ckpt_path)
# model.thresholds = dict(prob=prob_threshold, nms=0.4)
# print("Now the thresholds are:", model.thresholds)

# Predict on the whole slide image (wsi) or a test image
if wsi_filepath:
    slide = openslide.open_slide(wsi_filepath)
    patches = DeepZoomGenerator(slide, tile_size=patch_len, overlap=0)
    last_level = patches.level_count - 1 # last level has the highest resolution
    cols, rows = patches.level_tiles[last_level]
    whole_mask = np.zeros((rows*patch_len, cols*patch_len))

    for i in range(0, cols):
        for j in range(0, rows):
            patch = patches.get_tile(last_level, (i, j))
            patch = np.array(patch)
            patch = normalize_mi_ma(patch, mi=0, ma=255)
            patch = resize(patch, (target_len, target_len), anti_aliasing=False).astype(np.float32)
            pred_mask = (model.predict_instances(patch, prob_thresh=prob_threshold, scale=scale)[0] > 0).astype(np.uint8)
            pred_mask = Image.fromarray(pred_mask * 255, mode='L')
            pred_mask = pred_mask.resize((patch_len, patch_len))
            pred_mask = np.array(pred_mask)
            whole_mask[j*patch_len:(j+1)*patch_len, i*patch_len:(i+1)*patch_len] = pred_mask

    whole_mask = (whole_mask > 0).astype(np.uint8)
    whole_mask = Image.fromarray(whole_mask * 255)
    whole_mask = whole_mask.convert('L')
    whole_mask.save(os.path.join(output_dir, f"pred_whole_mask.png"))
    whole_mask.thumbnail((1000, 1000))
    # whole_mask.show()
    whole_mask.save(os.path.join(output_dir, f"pred_whole_mask_thumbnail.png"))

elif test_image_filepath:
    test_image = io.imread(test_image_filepath)
    test_image = normalize_mi_ma(test_image, mi=0, ma=255)
    labels, details = model.predict_instances(test_image, prob_thresh=prob_threshold, scale=scale)
    coords, probs = details["coord"], details["prob"]
    np.save(os.path.join(output_dir, 'pred_labels.npy'), labels)
    np.save(os.path.join(output_dir, 'pred_coords.npy'), coords)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    lbl_cmap = random_label_cmap()
    ax.imshow(test_image, alpha=0.7)
    ax.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    ax.axis('off')
    fig.savefig(os.path.join(output_dir, f"pred_whole_mask.png"),
                bbox_inches='tight', dpi=300, transparent=True)

else:
    raise ValueError("Either wsi_filepath or test_image_filepath must be provided.")

end = time()
print(f"Time elapsed: {(end-start)/60:.2f} minutes.")