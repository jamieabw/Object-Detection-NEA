import os
import json
from PIL import Image

def convert_odgt_to_darknet(annotation_file, images_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the .odgt file line by line
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Parse each line as a JSON object
        annotation = json.loads(line.strip())

        # Extract image details
        image_id = annotation['ID']
        image_path = os.path.join(images_folder, image_id + '.jpg')

        # Load image to get its dimensions
        try:
            image = Image.open(image_path)
            img_width, img_height = image.size
        except FileNotFoundError:
            print(f"Image {image_id} not found, skipping...")
            continue

        # Prepare the text file path
        txt_file_path = os.path.join(output_folder, image_id + '.txt')

        with open(txt_file_path, 'w') as txt_file:
            for bbox in annotation['gtboxes']:
                # Only consider 'human' class (if applicable)
                if bbox['tag'] != 'person':
                    continue

                # Get the bounding box coordinates
                x1, y1, w, h = bbox['vbox']
                x_center = x1 + w / 2.0
                y_center = y1 + h / 2.0

                # Normalize the coordinates
                x_center /= img_width
                y_center /= img_height
                w /= img_width
                h /= img_height

                # Write to the file in Darknet format: <class> <x_center> <y_center> <width> <height>
                txt_file.write(f"0 {x_center} {y_center} {w} {h}\n")
# Usage
annotation_file = 'C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset - CROWDHUMAN\\annotation_train.odgt'
images_folder = 'C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\dataset - CROWDHUMAN\\CrowdHuman_train'
output_folder = 'C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\dataset - CROWDHUMAN\\CrowdHuman_train'
convert_odgt_to_darknet(annotation_file, images_folder, output_folder)
