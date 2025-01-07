import os
from PIL import Image, ImageEnhance

def augment_images(input_folder, output_folder):
    """
    Augments images in the specified input folder and saves them to the output folder.

    Args:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path to the folder to save augmented images.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Open the image if it is a valid file
        try:
            with Image.open(file_path) as img:
                # Ensure image is in RGB mode
                img = img.convert("RGB")

                # Perform augmentations
                augmentations = [
                    ("_rot90", img.rotate(90)),
                    ("_rot180", img.rotate(180)),
                    ("_rot270", img.rotate(270)),
                    ("_flip_h", img.transpose(Image.FLIP_LEFT_RIGHT)),
                    ("_flip_v", img.transpose(Image.FLIP_TOP_BOTTOM)),
                    ("_bright", ImageEnhance.Brightness(img).enhance(1.5)),
                    ("_dark", ImageEnhance.Brightness(img).enhance(0.5))
                ]

                # Save original and augmented images
                base_name, ext = os.path.splitext(filename)
                img.save(os.path.join(output_folder, f"{base_name}_original{ext}"))
                for suffix, augmented_img in augmentations:
                    augmented_img.save(os.path.join(output_folder, f"{base_name}{suffix}{ext}"))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Specify the input and output folders
input_folder = "/Users/othmaneirhboula/Downloads/negatif"
output_folder = "/Users/othmaneirhboula/Downloads/negatif_augmented"

# Run the augmentation
augment_images(input_folder, output_folder)
