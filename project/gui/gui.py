from tkinter import *
from tkinter import filedialog

from ..htr.preprocessor.image_augmentor import Augmentor

from PIL import ImageTk, Image
import cv2

class HtrGui:
    root = None
    model = None

    def __init__(self, model):
        assert model is not None
        self.root = Tk(className="HTR Gui")
        self.model = model

        self.set_geometry()
        self.add_controls()

        self.root.mainloop()

    def set_geometry(self):
        self.root.geometry("500x500+300+300")

    def add_controls(self):
        self.button_explore = Button(
            self.root,
            text="Wähle Bild mit Text",
            command=self.browse_files
        )
        self.button_explore.pack()

        self.original_image_label_result_text = StringVar()
        self.original_image_label = Label(self.root, textvariable=self.original_image_label_result_text)
        self.original_image_label.pack()

        self.image_original = Label(image=None)
        self.image_original.pack(fill=BOTH)

        self.augmented_image_label_result_text = StringVar()
        self.augmented_image_label = Label(self.root, textvariable=self.augmented_image_label_result_text)
        self.augmented_image_label.pack()
        self.image_augmented = Label(image=None)
        self.image_augmented.pack(fill=BOTH)


    def browse_files(self):
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Wähle Bild mit Text",
            filetypes=(
                ("Images", ".jpg .png"),
                ("all files", "*.*")
            )
        )

        if len(filename) == 0:
            return

        cv_image = cv2.imread(filename, 0)
        augmented = Augmentor.preprocess(img=cv_image, image_size=(128, 32), augment=False)

        img = Image.fromarray(augmented * 255)

        to_predict = ImageTk.PhotoImage(img)

        original = Image.open(filename)
        # Target Width = 200
        # ImageW * x = 200
        # 200 / ImageW = x
        x = 200 / original.size[0]
        original = original.resize((round(original.size[0]*x), round(original.size[1] * x)))
        original = ImageTk.PhotoImage(original)

        self.image_augmented.configure(image=to_predict)
        self.image_augmented.image = to_predict

        self.image_original.configure(image=original)
        self.image_original.image = original

        self.original_image_label_result_text.set("Eingabebild:")
        self.augmented_image_label_result_text.set("Eingabebild in das NN:")

        self.root.update_idletasks()
