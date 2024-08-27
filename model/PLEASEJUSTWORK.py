from dataProcessing import preprocessData
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

def display_image_with_bbox(image_path, x_centre, y_centre, width, height):
    # Load the image
    image = Image.open(image_path).resize((448,448))
    img_width, img_height = image.size
    # Convert normalized coordinates to pixel coordinates
    # Calculate the bounding box coordinates
    x_min = x_centre - (width / 2)
    y_min = y_centre - (height / 2)
    x_max = x_centre + (width / 2)
    y_max = y_centre + (height / 2)

    # Display the image with the bounding box
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Create a Rectangle patch for the bounding box
    rect = patches.Rectangle((x_min, y_min), width, height,
                              edgecolor='red', facecolor="none")
    
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    # Show the image
    plt.show()


labeldir = "model\\testdata\\crop_000003_jpg.rf.df647b1de3032ae2eff2f38159b2be1c.txt"
imgdir = "model\\testdata\\crop_000003_jpg.rf.df647b1de3032ae2eff2f38159b2be1c.jpg"

#x_test, y_test = preprocessData("model\\testdata")
#print(x_test.shape, y_test.shape)
C = 448 / 8

def display_box(output, a, b):

    for j in range(7, -1, -1):
        print("\n")
        for i in range(8):
            #print(output[0,i,j], end=" ")
            if i == a and j == b:
                x = (C * i) + (C * output[0,i,j,1])
                y = (C * j) + (C * output[0,i,j,2])
                w = C * output[0,i,j,3]
                h = C * output[0,i,j,4]
                print(x, y, w, h)
                display_image_with_bbox(imgdir, x, y, w, h)

def results(output, a, b):
    display_box(output, a, b)


#results()