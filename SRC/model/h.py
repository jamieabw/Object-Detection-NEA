import os

def delete_empty_text_and_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_path = os.path.join(directory, filename)
            if os.path.getsize(txt_path) == 0:
                base_name = os.path.splitext(txt_path)[0]
                jpg_path = base_name + ".jpg"
                
                os.remove(txt_path)
                print(f"Deleted empty text file: {txt_path}")
                
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
                    print(f"Deleted associated image: {jpg_path}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    if os.path.isdir(directory):
        delete_empty_text_and_images(directory)
    else:
        print("Invalid directory path.")
