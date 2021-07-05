import os
import cv2
import numpy as np

from PIL import Image

names = []
paths = []

for users in os.listdir("face_images"):
    names.append(users)

for name in names:
    for image in os.listdir(f'face_images\{name}'):
        path_string = os.path.join(f'face_images\{name}', image)
        paths.append(path_string)
print(paths)

faces = []
ids = []

for img_path in paths:
    image = Image.open(img_path).convert("L")
    imgNp = np.array(image, "uint8")

    faces.append(imgNp)
    id = int(img_path.split("\\")[2].split("_")[0])
    print(id)
    ids.append(id)

ids = np.array(ids)
#pip install opencv-contrib-python
trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(faces, ids)
trainer.write("training.yml")

