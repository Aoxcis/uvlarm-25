import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Charger le modèle SVM
svm = joblib.load('svm_model.pkl')


# Fonction pour extraire les caractéristiques HOG
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features


# Charger l'image à tester
image_path = '23.png'
image = cv2.imread(image_path)

if image is None:
    print("Erreur lors du chargement de l'image.")
else:
    # Extraire les caractéristiques HOG de l'image
    features = extract_hog_features(image)

    # Redimensionner les caractéristiques si nécessaire (en fonction de l'entraînement)
    features = features.reshape(1, -1)

    # Faire la prédiction avec le modèle SVM
    prediction = svm.predict(features)

    # Afficher le résultat
    if prediction == 1:
        print("L'image est classée comme positive.")
    else:
        print("L'image est classée comme négative.")
