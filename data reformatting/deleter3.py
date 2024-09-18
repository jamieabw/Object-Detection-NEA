import os

# Directory containing files to be deleted
directory_path = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file name starts with 'crop'
    if filename.startswith('person'):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
