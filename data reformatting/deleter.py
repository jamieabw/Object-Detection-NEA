import os

def clean_dataset(images_folder, labels_folder, image_extensions=['.jpg', '.jpeg', '.png']):
    # Get all image and label files
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_folder) 
                   if os.path.splitext(f)[1].lower() in image_extensions}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) 
                   if f.endswith('.txt')}
    
    # Find images without corresponding labels
    images_without_labels = image_files - label_files
    # Find labels without corresponding images
    labels_without_images = label_files - image_files

    # Delete images without labels
    for image_name in images_without_labels:
        for ext in image_extensions:
            image_path = os.path.join(images_folder, image_name + ext)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image without label: {image_path}")

    # Delete labels without images
    for label_name in labels_without_images:
        label_path = os.path.join(labels_folder, label_name + '.txt')
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted label without image: {label_path}")

# Usage
images_folder = 'C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\training data set'
labels_folder = 'C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\training data set'
clean_dataset(images_folder, labels_folder)
