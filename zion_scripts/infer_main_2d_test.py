# %%
import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize
from csbdeep.utils import normalize_mi_ma
import matplotlib.pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator
from stardist.models import Config2D, StarDist2D
from time import time

np.random.seed(42)
image_name          = "240819_Ji_D1_H_EScan"
model_ckpt_path     = "/hpc/group/yizhanglab/zs144/repo/Zion-stardist/examples/2D/models/models_v0.9/"
wsi_filepath        = f"/hpc/group/yizhanglab/DATA/zs144/MedSAM/neuron_wsi/{image_name}.tif"
test_image_filepath = None
patch_len           = 2048
target_len          = 256
prob_threshold      = 0.2
output_dir          = f"/hpc/group/yizhanglab/zs144/Zion-ZhangLab/projects/segment_neurons/output/{image_name}/pred_whole_mask/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start = time()

# Set configuration
conf = Config2D (
    n_rays       = 32,
    grid         = (4, 4),
    use_gpu      = True,
    n_channel_in = 3, # RGB
    train_patch_size = (target_len, target_len)
)

vars(conf)

# Load pretrained model
model = StarDist2D(None, name='stardist', basedir=model_ckpt_path)
# model.thresholds = dict(prob=prob_threshold, nms=0.4)
print("Now the thresholds are:", model.thresholds)
#%%
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob
from tqdm import tqdm
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from skimage.transform import resize
from csbdeep.utils import Path, normalize_mi_ma
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

X = sorted(glob('/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP004/output/240819_Ji_N1_H_EScan/patches_2048/*.png'))
Y = sorted(glob('/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP004/output/240819_Ji_N1_H_EScan/pseudo_gt_masks_2048/*.png'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

X = [io.imread(x) for x in X]
Y = [io.imread(y) for y in Y]
Y = [(mask // 255).astype(np.uint8) for mask in Y]
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (0,1) # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize_mi_ma(x, 0, 255) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]
# resize image
target_size = (256, 256)
X = [resize(x, target_size, anti_aliasing=False).astype(np.float32) for x in tqdm(X)]
Y = [(resize(y, target_size, anti_aliasing=False) > 0).astype(np.uint8) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
X_trn, X_test, Y_trn, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_test))

#%%
Y_test_pred = [(model.predict_instances(x, prob_thresh=0.2)[0] > 0).astype(np.uint8)
              for x in tqdm(X_test)]

def plot_mask_pred(img, mask, pred):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im = axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(mask * 255, cmap='gray')
    axes[1].set_title("Truth")
    axes[2].imshow(pred * 255, cmap='gray')
    axes[2].set_title("Pred")
    plt.tight_layout()

i = 1
plot_mask_pred(X_test[i], Y_test[i], Y_test_pred[i])

# %%
# Predict on the whole slide image (wsi) or a test image
slide = openslide.open_slide(wsi_filepath)
patches = DeepZoomGenerator(slide, tile_size=patch_len, overlap=0)
last_level = patches.level_count - 1 # last level has the highest resolution
cols, rows = patches.level_tiles[last_level]
whole_mask = np.zeros((rows*patch_len, cols*patch_len))

# for i in range(0, cols):
#     for j in range(0, rows):
i = 3
j = 4
patch = patches.get_tile(last_level, (i, j))
patch = np.array(patch)
patch = normalize_mi_ma(patch, mi=0, ma=255)
patch = resize(patch, (target_len, target_len), anti_aliasing=False).astype(np.float32)
# show the patch
plt.imshow(patch)

#%%
pred_mask = (model.predict_instances(patch, prob_thresh=0.2)[0] > 0).astype(np.uint8)
pred_mask = Image.fromarray(pred_mask * 255, mode='L')
pred_mask.show()
pred_mask = pred_mask.resize((patch_len, patch_len))
pred_mask = np.array(pred_mask)
# %%
whole_mask[j*patch_len:(j+1)*patch_len, i*patch_len:(i+1)*patch_len] = pred_mask

whole_mask = (whole_mask > 0).astype(np.uint8)
whole_mask = Image.fromarray(whole_mask * 255)
whole_mask = whole_mask.convert('L')
# whole_mask.save(os.path.join(output_dir, f"pred_whole_mask.png"))
whole_mask.thumbnail((1000, 1000))
whole_mask.show()
# whole_mask.save(os.path.join(output_dir, f"pred_whole_mask_thumbnail.png"))

end = time()
print(f"Time elapsed: {(end-start)/60:.2f} minutes.")
# %%
