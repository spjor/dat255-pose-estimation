import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
"from Alek import resnet"


class PoseModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")

        "self. model = resnetblock()"
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.ToTensor()
        self.person_score_threshold = 0.7

        self.skeleton = [
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (11, 12),
            (5, 11), (6, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (0, 1), (0, 2),
            (1, 3), (2, 4),
        ]

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            output = self.model([img_tensor])[0]

        return image, output

    def draw_pose(self, image, output):
        draw = ImageDraw.Draw(image)

        boxes = output["boxes"].detach().cpu()
        scores = output["scores"].detach().cpu()
        keypoints = output["keypoints"].detach().cpu()

        for i in range(len(scores)):
            if scores[i] < self.person_score_threshold:
                continue

            box = boxes[i].tolist()
            kps = keypoints[i]

            draw.rectangle(box, outline="red", width=3)

            for a, b in self.skeleton:
                xa, ya, _ = kps[a].tolist()
                xb, yb, _ = kps[b].tolist()
                draw.line((xa, ya, xb, yb), fill="lime", width=3)

            for kp in kps:
                x, y, _ = kp.tolist()
                r = 4
                draw.ellipse((x - r, y - r, x + r, y + r), fill="yellow", outline="black")

        return image

    def analyze_image(self, image_path):
        image, output = self.predict(image_path)
        result = image.copy()
        return self.draw_pose(result, output)