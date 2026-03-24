import keras
import tensorflow
import numpy
import matplotlib
import fiftyone as fo
import fiftyone.zoo as foz


if fo.dataset_exists("coco-2017-validation-50"):
    fo.delete_dataset("coco-2017-validation-50")

name = "coco-2017"

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["keypoints"],   # or ["segmentations"], etc, depending on what you want
    classes=["person"],
    max_samples=50,
)


# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())
# Load directly from the zoo (changes are ephemeral)
#dataset = foz.load_zoo_dataset("quickstart-groups")

print(f"Dataset: {dataset.name}")
print(f"Media type: {dataset.media_type}")
print(f"Group slices: {dataset.group_slices}")
print(f"Default slice: {dataset.default_group_slice}")
#print(f"Num groups (scenes): {len(dataset.distinct('group.id'))}")

#group_ids = dataset.distinct("group.id")
#print(f"Total groups (scenes): {len(group_ids)}")

# Examine first group
#example_group = dataset.get_group(group_ids[0])
#print(f"\nSamples in first group:")
#for slice_name, sample in example_group.items():
    #print(f"  {slice_name}: {sample.filepath.split('/')[-1]}")
# Visualize the dataset in the FiftyOne App
session = fo.launch_app(dataset)

"""
# Fjerne korrupte filer, tatt rett fra notebooks
num_skipped = 0
for folder_name in ("keypoints"):
    folder_path = os.path.join("coco-2017", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")
"""
"""
# gjøre om filer til tall✨

# dette er på ingen måte rett, dette må endres på
def create_pose_model(num_keypoints=17):
    inputs = keras.Input(shape=(180, 180, 3))

    # Backbone
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    # Global features
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    # Output: (x, y) coordinates for each keypoint
    outputs = keras.layers.Dense(num_keypoints * 2)(x)  # No activation
    outputs = keras.layers.Reshape((num_keypoints, 2))(outputs)

    return keras.Model(inputs, outputs)


model = create_pose_model()


def preprocess_keypoints(image, keypoints, image_size=(180, 180)):
    # Resize image
    image = tf.image.resize(image, image_size)

    # Normalize keypoints to [0, 1] based on image dimensions
    # Assuming keypoints are in format [[x1,y1], [x2,y2], ...]
    keypoints = keypoints / tf.cast(tf.shape(image)[:2], tf.float32)

    return image, keypoints

model.compile(
    optimizer='adam',
    loss='mse',  # Mean squared error for coordinate regression
    metrics=['mae']  # Mean absolute error
)

# Custom callback to visualize predictions
class VisualizePredictions(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Visualize some predictions
        pass

#todo: trene modellen
#todo: teste modellen
#todo: datapipleline
#todo: loss function
#todo: data augmentation
#todo: Evaluation metrics
#todo: visualtisation
#todo: post-processing, hva som ligger i dette? ikke spør meg. Men det er tydeligvis noe vi burde gjøre

"""