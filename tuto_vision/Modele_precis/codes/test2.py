import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Charger le modèle SVM
svm_model = joblib.load("svm_model.pkl")

# Fonction pour extraire des caractéristiques HOG sur chaque canal de couleur
def extract_hog_features_color(image):
    # Séparer les canaux de couleur
    channels = cv2.split(image)
    features = []
    for channel in channels:
        # Calculer les caractéristiques HOG pour chaque canal
        resized_channel = cv2.resize(channel, (200, 150))  # Taille utilisée à l'entraînement
        channel_hog, _ = hog(resized_channel, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.extend(channel_hog)
    return np.array(features)

# Fonction pour détecter les objets
def detect_objects(frame, svm_model, window_size=(200, 150), step_size=32):
    detected_boxes = []
    h, w = frame.shape[:2]

    # Redimensionner pour accélérer (facultatif)
    resized_frame = cv2.resize(frame, (640, 480))  # Ajuster selon la puissance du PC
    scale_x = w / 640
    scale_y = h / 480

    # Glisser une fenêtre
    for y in range(0, resized_frame.shape[0] - window_size[1], step_size):
        for x in range(0, resized_frame.shape[1] - window_size[0], step_size):
            window = resized_frame[y:y + window_size[1], x:x + window_size[0]]
            resized_window = cv2.resize(window, (200, 150))  # Taille utilisée à l'entraînement
            features = extract_hog_features_color(resized_window).reshape(1, -1)

            # Utilisation de la fonction de décision du SVM pour obtenir la probabilité
            prediction = svm_model.decision_function(features)

            # Si la probabilité dépasse le seuil (par exemple, 0.7), considérer cela comme une détection
            if prediction > 0.999999999:
                detected_boxes.append((
                    int(x * scale_x),
                    int(y * scale_y),
                    int(window_size[0] * scale_x),
                    int(window_size[1] * scale_y)
                ))

    return detected_boxes

# Capturer la vidéo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

print("Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture vidéo.")
        break

    # Détecter les objets
    boxes = detect_objects(frame, svm_model)

    # Dessiner les boîtes détectées
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher la vidéo
    cv2.imshow("Détection d'objets", frame)

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
