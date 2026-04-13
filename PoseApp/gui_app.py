import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from PoseModel import PoseModel


class GuiApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x800")
        self.root.title("Pose Estimation")
        self.root.config(cursor="gumby")

        self.model = PoseModel()

        self.status_label = tk.Label(root, text="Velkomen! Velg et bilde")
        self.status_label.pack(pady=20)

        self.button = tk.Button(root, text="Velg bilde", command=self.open_image)
        self.button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        self.status_label.config(text="Bilde valgt! Kjører analyse...")
        self.root.update_idletasks()

        try:
            result_img = self.model.analyze_image(file_path)

            display_img = result_img.copy()
            display_img.thumbnail((700, 700))

            img_tk = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            self.status_label.config(text="Analyse ferdig.")

        except Exception as e:
            self.status_label.config(text=f"Feil: {e}")
            print(e)


if __name__ == "__main__":
    root = tk.Tk()
    app = GuiApp(root)
    root.mainloop()