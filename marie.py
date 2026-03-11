import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import fiftyone
import fiftyone.zoo as foz

# ✨Marie prøver seg fram ✨

# laste ned datasettet, vet ikke om fifty one er en god ide
dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["person", "car"],
    max_samples=50,
)

# Visualize the dataset in the FiftyOne App, laste ned data settet
session = fiftyone.launch_app(dataset)

# visualisere, se om jeg får ut head
# Fjerne korrupte filer
# gjøre om filer til tall✨
# er data augmentation nødvendig? stoort datasett men idk, er det noen gang stort nok?



image_shape = (180, 180, 3)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages", # endre
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_shape[:2],
    batch_size=batch_size,
)

# sequential model basic greier tatt rett fra notebooks
sequential_model = keras.Sequential(
    [
        keras.Input(shape=image_shape),
        keras.layers.Rescaling(1.0/255),    # Standardise the images on-the-fly
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
