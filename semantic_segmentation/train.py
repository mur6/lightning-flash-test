from functools import partial

import flash

# from flash.core.utilities.imports import example_requires
from flash.image import InstanceSegmentation, InstanceSegmentationData


data_dir = "/Users/taichi.muraki/workspace/programming/DataSynthesis/blenderproc/data/outputs/coco3/1"
train_ann_file = f"{data_dir}/coco_annotations.json"


datamodule = InstanceSegmentationData.from_coco(
    train_folder=data_dir,
    train_ann_file=train_ann_file,
    val_split=0.1,
    batch_size=4,
)

# 2. Build the task
model = InstanceSegmentation(
    head="mask_rcnn",
    backbone="resnet18_fpn",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Detect objects in a few images!
# datamodule = InstanceSegmentationData.from_files(
#     predict_files=[
#         str(data_dir / "images/yorkshire_terrier_9.jpg"),
#         str(data_dir / "images/yorkshire_terrier_12.jpg"),
#         str(data_dir / "images/yorkshire_terrier_13.jpg"),
#     ],
#     batch_size=4,
# )
# predictions = trainer.predict(model, datamodule=datamodule)
# print(predictions)

# 5. Save the model!
trainer.save_checkpoint("instance_segmentation_model.pt")
