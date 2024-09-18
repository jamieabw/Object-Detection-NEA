import os

# Directory containing YOLOv1 annotations and images
annotations_dir = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\PASCAL VOC\\valid\\VOCdevkit\\VOC2007\\JPEGImages"
images_dir = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\PASCAL VOC\\valid\\VOCdevkit\\VOC2007\\JPEGImages"

# Function to check if at least one line starts with the number 14


# Iterate through all annotation files in the directory
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith('.txt'):
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        # Filter out lines that don't start with 14 and replace 14 with 0
        modified_lines = []
        for line in lines:
            if line.strip() and line.split()[0] == '14':
                # Replace '14' with '0' at the beginning of the line
                modified_line = line.replace('14', '0', 1)
                modified_lines.append(modified_line)
        
        # Write the modified lines back to the file
        with open(annotation_path, 'w') as file:
            file.writelines(modified_lines)
        
        # If no lines remain, delete the file
        if not modified_lines:
            os.remove(annotation_path)
            print(f"Deleted empty annotation file: {annotation_path}")
