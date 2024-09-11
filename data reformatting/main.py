import os
import xml.etree.ElementTree as ET

INPUT_DIR = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\Annotations" 
OUTPUT_DIR = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\YOLOLabels"

# Class list in the same order as PASCAL VOC
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def convert_bbox(size, box):
    """Convert bounding box to YOLO format (x_center, y_center, width, height)"""
    dw = 1.0 / size[0]  # image width
    dh = 1.0 / size[1]  # image height
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".xml"):
        # Parse the XML
        in_file = os.path.join(INPUT_DIR, file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        # Image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Output label file (same name as XML, but .txt)
        out_file = os.path.join(OUTPUT_DIR, file.replace(".xml", ".txt"))
        with open(out_file, 'w') as f:
            for obj in root.iter('object'):
                # Get class label
                class_name = obj.find('name').text
                if class_name in classes:
                    class_id = classes.index(class_name)
                    
                    # Get bounding box
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bbox = convert_bbox((width, height), b)
                    
                    # Write to YOLO format
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")