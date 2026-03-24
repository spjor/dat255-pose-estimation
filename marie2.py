
import fiftyone as fo
import fiftyone.zoo as foz

def main():
    dataset_name = "coco-2017-validation-50"

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["person"],
        only_matching=True,
        max_samples=50,
        shuffle=True,
        seed=51,

    )

    model = foz.load_zoo_model("keypoint-rcnn-resnet50-fpn-coco-torch")
    dataset.default_skeleton = model.skeleton

    dataset.apply_model(
        model,
        label_field="pred_keypoints",
        num_workers=0,   # important on your setup
    )

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
