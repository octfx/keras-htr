import os
from tkinter import *
from tkinter import filedialog

import cv2
from PIL import ImageTk, Image

from ..htr.models import predict
from ..htr.preprocessor.image_augmentor import Augmentor


class HtrGui:
    root = None
    model = None
    image_input = None

    def __init__(self, model_path):
        assert os.path.exists(model_path)

        self.root = Tk(className="HTR Gui")
        self.root.wm_title("HTR Gui")
        self._model_path = model_path

        self.set_geometry()
        self.add_controls()

        self.root.mainloop()

    def set_geometry(self):
        self.root.geometry("500x500+300+300")

    def add_controls(self):
        self.button_explore = Button(
            self.root,
            text="Select image with handwritten text",
            command=self.browse_files
        )
        self.button_explore.pack()

        Label(self.root, text="Decoding mode:").pack()

        self.decode_value = StringVar(self.root)
        self.decode_value.set("Greedy")
        modes = ["Greedy", "Beam Search"]
        self.decode_select = OptionMenu(self.root, self.decode_value, *modes)
        self.decode_select.pack()

        self.original_image_label_result_text = StringVar(self.root)
        self.original_image_label = Label(self.root, textvariable=self.original_image_label_result_text)
        self.original_image_label.pack()

        self.image_original = Label(image=None)
        self.image_original.pack(fill=BOTH)

        self.augmented_image_label_result_text = StringVar(self.root)
        self.augmented_image_label = Label(self.root, textvariable=self.augmented_image_label_result_text)
        self.augmented_image_label.pack()
        self.image_augmented = Label(image=None)
        self.image_augmented.pack(fill=BOTH)

        self.status_text_content = StringVar(self.root)
        self.status_text = Label(self.root, textvariable=self.status_text_content)
        self.status_text.pack()

        self.result_text_content = StringVar(self.root)
        self.result_text = Label(self.root, textvariable=self.result_text_content)
        self.result_text.pack()

        self.rerun = Button(
            self.root,
            text="Re-Run Decoding",
            command=self.rerun
        )
        self.rerun.pack()

    def rerun(self):
        if self.image_input is None:
            return

        self.htr()

    def browse_files(self):
        self.status_text_content.set("")
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select image with handwritten text",
            filetypes=(
                ("Images", ".jpg .png"),
                ("all files", "*.*")
            )
        )

        if len(filename) == 0:
            return

        cv_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if cv_image is None:
            self.status_text_content.set("Error readint the selected image!")
            return

        augmented = Augmentor.preprocess(img=cv_image, image_size=(128, 32), augment=False, binarize=True)

        self.image_input = augmented

        img = Image.fromarray(augmented * 255)

        to_predict = ImageTk.PhotoImage(img)

        original = Image.open(filename)
        # Target Width = 128
        # ImageW * x = 128
        # 128 / ImageW = x
        x = 128 / original.size[0]
        original = original.resize((round(original.size[0] * x), round(original.size[1] * x)))
        original = ImageTk.PhotoImage(original)

        self.image_augmented.configure(image=to_predict)
        self.image_augmented.image = to_predict

        self.image_original.configure(image=original)
        self.image_original.image = original

        self.original_image_label_result_text.set("Input image:")
        self.augmented_image_label_result_text.set("Input image into the network:")

        self.root.update()
        self.root.update_idletasks()
        self.htr()

    def htr(self):
        self.status_text_content.set("Detecting text...")
        self.result_text_content.set("")
        self.root.update()
        self.root.update_idletasks()

        decode_mode = self.decode_value.get()
        if decode_mode == "Beam Search":
            decode_mode = "Beam"
        else:
            decode_mode = "Greedy"

        res = predict(
            model_path=self._model_path,
            char_table=os.path.join(os.getcwd(), 'data', 'characters.txt'),
            image=self.image_input,
            decode_mode=decode_mode
        )

        print('Recognized text ({}): "{}"'.format(decode_mode, res))

        self.status_text_content.set("Recognized text:")
        self.result_text_content.set(res)
        self.root.update()
        self.root.update_idletasks()
