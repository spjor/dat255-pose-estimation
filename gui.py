import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
model.to(device)

transform = transforms.ToTensor()

PERSON_SCORE_THRESHOLD = 0.7

SKELETON = [
    (5, 6),      # shoulders
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),    # hips
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
        draw.rectangle(box, outline="red", width=3)

        # Draw skeleton lines
        for a, b in SKELETON:
            xa, ya, _ = kps[a].tolist()
            xb, yb, _ = kps[b].tolist()
            draw.line((xa, ya, xb, yb), fill="lime", width=3)

        # Draw keypoints
        for kp in kps:
            x, y, _ = kp.tolist()
            r = 4
            draw.ellipse((x-r, y-r, x+r, y+r), fill="yellow", outline="black")

    return image


def analyze_image(file_path):
    image = Image.open(file_path).convert("RGB")
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

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

status_label = tk.Label(root, text="Velkomen! Velg et bilde")
status_label.pack(pady=20)

button = tk.Button(root, text="Velg bilde", command=open_image)
button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=20)

root.mainloop()