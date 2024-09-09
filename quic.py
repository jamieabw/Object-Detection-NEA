import os

# Directory path where your jpg and txt files are located
directory = 'C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\first training session set - Copy'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .jpg file
    if filename.endswith('.jpg'):
        # Get the file name without extension
        file_base = os.path.splitext(filename)[0]
        
        # Construct the corresponding .txt file name
        txt_file = file_base + '.txt'
        
        # Check if the .txt file exists
        if not os.path.exists(os.path.join(directory, txt_file)):
            # If the .txt file does not exist, delete the .jpg file
            os.remove(os.path.join(directory, filename))
            print(f"Deleted {filename} because the corresponding .txt file was not found.")
