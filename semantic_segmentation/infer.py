import torch

import flash
from flash import Trainer
from flash.image import SemanticSegmentation, SemanticSegmentationData


model = SemanticSegmentation.load_from_checkpoint("models/model-b4-ep1600.pt")

datamodule = SemanticSegmentationData.from_files(
    predict_files=["data/samples/5.jpeg", "data/samples/with_hand.jpeg"], batch_size=2
)
trainer = Trainer()

from flash.core.integrations.fiftyone import visualize

predictions = trainer.predict(model, datamodule=datamodule, output="fiftyone")
session = visualize(predictions, wait=True)

# predictions = trainer.predict(model, datamodule=datamodule)
# prediction = predictions[0]
# print(prediction)


# from flash.core.integrations.fiftyone import visualize

# predictions = trainer.predict(model, datamodule=datamodule, output="fiftyone")
# session = visualize(predictions, wait=True)


# # 3. Create the trainer and finetune the model
# trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
# trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# # 4. Segment a few images!
# datamodule = SemanticSegmentationData.from_files(
#     predict_files=[
#         "data/CameraRGB/F61-1.png",
#         "data/CameraRGB/F62-1.png",
#         "data/CameraRGB/F63-1.png",
#     ],
#     batch_size=3,
# )
# predictions = trainer.predict(model, datamodule=datamodule)
# print(predictions)
