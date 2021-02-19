# Keras neural network for handwritten text recognition
Project for the course "Industrial image processing" at Ostfalia University Wolfenbüttel in the winter term 2020/2021.

Activate the conda environment:
```shell
conda env create -f env.yml
conda env activate ibv-ocr
```

## Training
This requires the presence of the IAM dataset. You can download it at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
Download `words.tgz` and extract it into the `data` folder.

```shell
python train.py --model_path=model --learning_rate=0.001 --augment=true --epochs=100 --batch_size=150
```

On the first run, all images are loaded into a lmdb for faster access times.

## Examples
```shell
python htr.py model data\characters.txt examples\are.png
# -> are

python htr.py model data\characters.txt examples\bild.png
# -> Bild

python htr.py model data\characters.txt examples\wind.png
# -> wind (LER 0.25 w instead of W)

python htr.py model data\characters.txt examples\wort.png
# -> work (LER 0.25 k instead of t)
```

## Gui
Start the gui with

```shell
python main.py
```

## CLI Inference 
```shell
python htr.py model data\characters.txt examples\wind.png
```
