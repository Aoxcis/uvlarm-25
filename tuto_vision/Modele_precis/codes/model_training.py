import cv2
import numpy as np
import os
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import joblib

# Répertoires des images positives et négatives
positive_dir = "/Users/othmaneirhboula/Downloads/positif_final"
negative_dir = "/Users/othmaneirhboula/Downloads/negatif_final"

# Liste des fichiers dans les répertoires
positive_images = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.jpg')]
negative_images = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.jpg')]

# Préparer les labels (1 pour positif, 0 pour négatif)
labels = [1] * len(positive_images) + [0] * len(negative_images)

# Fonction pour extraire des caractéristiques HOG sur chaque canal de couleur
def extract_hog_features_color(image):
    # Séparer les canaux de couleur
    channels = cv2.split(image)
    features = []
    for channel in channels:
        # Calculer les caractéristiques HOG pour chaque canal
        resized_channel = cv2.resize(channel, (200, 150))  # Taille fixe pour l'entraînement
        channel_hog, _ = hog(resized_channel, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.extend(channel_hog)
    return np.array(features)

# Extraire les caractéristiques HOG pour chaque image
data = []
for image_path in positive_images + negative_images:
    img = cv2.imread(image_path)
    if img is not None:
        features = extract_hog_features_color(img)
        data.append(features)

# Convertir les caractéristiques en tableau numpy
X = np.array(data)
y = np.array(labels)

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un classificateur SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Sauvegarder le modèle entraîné
joblib.dump(svm, 'svm_model.pkl')

# Évaluer le modèle
print("Précision du modèle : ", svm.score(X_test, y_test))
