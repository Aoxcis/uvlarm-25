import os
from PIL import Image

def resize_and_rename_images(input_folder, output_folder):
    """
    Resizes images in the specified input folder to 800x600 and saves them
    in the output folder with sequential filenames.

    Args:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path to the folder to save resized images.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a counter for renaming
    counter = 1

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Open the image if it is a valid file
        try:
            with Image.open(file_path) as img:
                # Resize the image to 800x600
                img_resized = img.resize((800, 600))

                # Save the resized image with a new sequential name
                new_filename = f"{counter}.jpg"
                img_resized.save(os.path.join(output_folder, new_filename))
                counter += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Specify the input and output folders
input_folder = "/Users/othmaneirhboula/Downloads/positif_augmented"
output_folder = "/Users/othmaneirhboula/Downloads/positif_final"

# Run the resize and rename function
resize_and_rename_images(input_folder, output_folder)