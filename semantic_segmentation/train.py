import torch

import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData


# datamodule = SemanticSegmentationData.from_folders(
#     train_folder="data/CameraRGB",
#     train_target_folder="data/CameraSeg",
#     val_split=0.1,
#     transform_kwargs=dict(image_size=(256, 256)),
#     num_classes=21,
#     #
# )

datamodule = SemanticSegmentationData.from_folders(
    train_folder="custom_data/train_images/",
    train_target_folder="custom_data/train_masks/",
    val_split=0.2,
    transform_kwargs=dict(image_size=(256, 256)),
    num_classes=3,
    batch_size=4,
)

model = SemanticSegmentation(
    # backbone="mobilenetv3_large_100",
    # head="fpn",
    # backbone="xception",
    # head="unetplusplus",
    backbone="efficientnet-b2",
    head="unet",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=5, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

trainer.save_checkpoint("ss_model.pt")
