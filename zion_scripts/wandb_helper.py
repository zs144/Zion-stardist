import numpy as np
import wandb
from wandb.integration.keras import WandbEvalCallback
from sklearn.metrics import jaccard_score

from stardist.models import StarDistData2D

def wb_mask(bg_img, pred_mask, true_mask, labels):
    return wandb.Image(bg_img, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : labels},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : labels}
    })

# # fast.ai callback extension to log image masks
# class LogImagesCallback(Callback):

#     def __init__(self, model, images, labels, num_to_log=5):
#         self.model = model
#         self.images = images
#         self.labels = labels
#         self.num_to_log = num_to_log

#     def on_epoch_end(self, **kwargs):
#         if self.num_to_log > len(self.images):
#             print(f"Warning: num_to_log > num(images). Logging all {len(self.images)} images.")
#             self.num_to_log = len(self.images)
#         input_batch = self.images[:self.num_to_log]
#         mask_list = []
#         for i, img_pair in enumerate(input_batch):
#             original_image = img_pair[0]
#             # run the model on that image
#             pred_mask = self.model.predict_instances(original_image)[0]

#             # ground truth mask
#             true_mask = img_pair[1]
#             # keep a list of composite images
#             mask_list.append(wb_mask(original_image, pred_mask, true_mask, self.labels))

#         # log all composite images to W&B
#         wandb.log({"predictions" : mask_list})


class WandbSegEvalCallback(WandbEvalCallback):
    def __init__(
        self, model, validation_data, data_table_columns, pred_table_columns,
        labels, num_to_log=5
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.inner_model = model
        self.imgs = validation_data[0][:num_to_log]
        self.masks = validation_data[1][:num_to_log]
        self.labels = labels

    def add_ground_truth(self, logs=None):
        for idx, (img, true_mask) in enumerate(zip(self.imgs, self.masks)):
            self.data_table.add_data(idx, wandb.Image(img, masks={
                "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels}
            }))

    def add_model_predictions(self, epoch, logs=None):
        for idx, (img, true_mask) in enumerate(zip(self.imgs, self.masks)):
            # print("img shape: ", img.shape)
            # print("true_mask shape: ", true_mask.shape)
            pred_mask = (self.inner_model.predict_instances(img)[0] > 0).astype(np.uint8)
            # print("pred_mask shape: ", pred_mask.shape)
            iou = jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='binary')
            self.pred_table.add_data(
                epoch, idx, wb_mask(img, pred_mask, true_mask, self.labels), iou
            )
