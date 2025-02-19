import os
import sys
import numpy as np

from glob import glob
from tqdm import tqdm
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from skimage.transform import resize
from csbdeep.utils import Path, normalize_mi_ma

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D
from train_utils_2d import augmenter

np.random.seed(42)
lbl_cmap = random_label_cmap()


IMAGE_PATH = "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP004/output/240819_Ji_N1_H_EScan/patches_2048/"
MASK_PATH = "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP004/output/240819_Ji_N1_H_EScan/pseudo_gt_masks_2048/"
CKPT_PATH = "/hpc/group/yizhanglab/zs144/repo/Zion-stardist/zion_assets/checkpoints/"
TARGER_IMAGE_SIZE = (256, 256)
RANDOM_SEED = 42
N_RAYS = 32
USE_GPU = True
GRID_SIZE = (4, 4)
TRAIN_BATCH_SIZE = 4
TRAIN_LR = 3e-4
N_EPOCHS = 50

X = sorted(glob(os.path.join(IMAGE_PATH, '*.png')))
Y = sorted(glob(os.path.join(MASK_PATH, '*.png')))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
X = [io.imread(x) for x in X]
Y = [io.imread(y) for y in Y]
Y = [(mask // 255).astype(np.uint8) for mask in Y] # 0: background, 1: object
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# resize images and masks
X = [resize(x, TARGER_IMAGE_SIZE, anti_aliasing=False).astype(np.float32)
     for x in tqdm(X)]
Y = [(resize(y, TARGER_IMAGE_SIZE, anti_aliasing=False) > 0).astype(np.uint8)
     for y in tqdm(Y)]

# normalize images
X = [normalize_mi_ma(x, 0, 255) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

X_trn, X_test, Y_trn, Y_test = train_test_split(X, Y, test_size=0.15,
                                                random_state=RANDOM_SEED)
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_test))


conf = Config2D (
    n_rays       = N_RAYS,
    grid         = GRID_SIZE,
    use_gpu      = USE_GPU,
    n_channel_in = n_channel,
    train_patch_size = TARGER_IMAGE_SIZE,
    train_batch_size = TRAIN_BATCH_SIZE,
    train_learning_rate = TRAIN_LR
)

print(vars(conf))

model = StarDist2D(conf, name='stardist', basedir=CKPT_PATH)

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")


model.train(
    X_trn, Y_trn, validation_data=(X_test,Y_test),
    augmenter=augmenter, epochs=N_EPOCHS
)