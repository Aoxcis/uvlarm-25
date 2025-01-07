# Charger le fichier d'entrée
input_txt = "negatif.txt"
output_txt = "negatives_with_boxes.txt"

# Paramètres de l'image et de la boîte
image_width, image_height = 800, 800
box_width, box_height = 400, 400
x = (image_width - box_width) // 2
y = (image_height - box_height) // 2

# Lire le fichier existant et ajouter les informations de boîte
with open(input_txt, "r") as infile, open(output_txt, "w") as outfile:
    for line in infile:
        image_path = line.strip()  # Chemin de l'image
        # Ajouter les informations au format : chemin 1 x y largeur hauteur
        outfile.write(f"{image_path} 1 {x} {y} {box_width} {box_height}\n")
