import cv2
import os
import numpy as np

# Répertoire où sont stockées vos images positives
positive_dir = "/Users/othmaneirhboula/Downloads/positif_final"
positive_txt = "positives_with_boxes.txt"
output_vec = "samples.vec"

# Lire les images et créer des échantillons pour le fichier .vec
with open(positive_txt, "r") as file:
    images = file.readlines()

samples = []

for image_path in images:
    image_path = image_path.split()[0]
    img = cv2.imread(image_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray, (24, 24))
        samples.append(resized_img)

# Convertir les images en un format adapté pour .vec
samples = np.array(samples)
samples = samples.reshape((-1, 24 * 24))

# Créer le fichier .vec
cv2.imwrite(output_vec, samples)
print("Fichier samples.vec créé avec succès")
