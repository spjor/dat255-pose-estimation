import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import torch
import torchvision
from torchvision import transforms

#bruker gpuen hvis den er tilgjengelig, hvis ikke flytter den til cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#her er modellen, lager noe defualt greier med keypoint rcnn og ferdig trent restnet, disse vektene er trent på coco-2017
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")
model.eval() #model evaluation mode
print(model.eval())
model.to(device)


def model1():
    m = "her kommer modellen i guess"

"""
#dummy input for å teste modellen
dummy_input = [torch.randn(3, 224, 224).to(device)]
with torch.no_grad():
    prediction = model(dummy_input)

print("prediction:")
print(prediction)
print("prediction: ferdig")
"""

transform = transforms.ToTensor()

#viser bare detections med 70% sikkerhet at de er en ekte person
PERSON_SCORE_THRESHOLD = 0.7
# COCO keypoint names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
SKELETON = [
    (5, 6),      # skuldre
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),    # hofter
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
]

def draw_pose(image, output):
    draw = ImageDraw.Draw(image)

    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    keypoints = output["keypoints"].detach().cpu()

    for i in range(len(scores)):
        if scores[i] < PERSON_SCORE_THRESHOLD:
            continue


        box = boxes[i].tolist()
        kps = keypoints[i]

        # Draw box
        draw.rectangle(box, outline="red", width=3) #tegner boks

        # tegner skjellett
        for a, b in SKELETON:
            xa, ya, _ = kps[a].tolist()
            xb, yb, _ = kps[b].tolist()
            draw.line((xa, ya, xb, yb), fill="lime", width=3)

        #tegner keypoints
        for kp in kps:
            x, y, _ = kp.tolist()
            r = 4
            draw.ellipse((x-r, y-r, x+r, y+r), fill="yellow", outline="black")

    return image


def analyze_image(file_path):
    image = Image.open(file_path).convert("RGB")
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0] #her bruker vi modellen igjen

    result = image.copy()
    result = draw_pose(result, prediction)
    return result


def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    status_label.config(text="Bilde valgt! Kjører analyse...")
    root.update_idletasks()

    try:
        result_img = analyze_image(file_path)

        display_img = result_img.copy()
        display_img.thumbnail((700, 700))

        img_tk = ImageTk.PhotoImage(display_img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        status_label.config(text="Analyse ferdig.")
    except Exception as e:
        status_label.config(text=f"Feil: {e}")
        print(e)


root = tk.Tk()
root.geometry("800x800")
root.title("Pose Estimation")
root.config(cursor="gumby") #denne settningen er veldig viktig

status_label = tk.Label(root, text="Velkomen! Velg et bilde")
status_label.pack(pady=20)

button = tk.Button(root, text="Velg bilde", command=open_image)
button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=20)

root.mainloop()