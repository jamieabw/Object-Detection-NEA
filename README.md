# Object Detection NEA Project (A-Level CS NEA 2025)

This project was developed as part of my A-Level Computer Science NEA (Non-Exam Assessment) in 2025. It is a complete pipeline for object detection using a YOLOv1-type model, including training and real-time testing capabilities.

## Overview

The application allows you to:

- **Train a YOLOv1-style object detection model** on any custom dataset, as long as it is labeled in **YOLO-DARKNET format**. You can train the model to detect any objects you want.
- **Monitor detailed statistics during training**, including mean Average Precision (mAP) and loss graphs, to evaluate performance and track improvements.
- **Test the trained model** on images, video files, or live webcam footage to see real-time detections.
- **Adjust various settings** to customize how detections are displayed, including:
  - Border (bounding box) width
  - Border (bounding box) color
  - Detection confidence threshold
  - Webcam device selection (if multiple webcams are connected)

## Features

- Supports importing and training on custom datasets in YOLO-DARKNET format.
- Displays live feedback during training, helping you understand model behavior and progress.
- Flexible testing options: static images, pre-recorded videos, or live webcam feed.
- User-friendly interface with settings for visual customization and performance tuning.

## Tech Stack & Dependencies

This project uses the following libraries and frameworks:

- **TensorFlow** — core deep learning framework for training and inference
- **Keras** — high-level API for building and training the model
- **OpenCV** — for video and image processing
- **Tkinter** — to build the graphical user interface
- **NumPy** — for numerical operations
- **Pillow** — for image handling
- **Matplotlib** — for plotting training statistics
- **Numba** — to accelerate certain calculations

## Known Issues

This is a student research project, so there may be some bugs or edge cases that are not fully addressed. Testing has been done on a limited set of devices and configurations, so certain setups might require additional adjustments.

## Getting Started

1. Prepare your labeled dataset in **YOLO-DARKNET format**.
2. Train the model using the provided training interface and monitor mAP and loss graphs in real time.
3. Test the trained model on images, video files, or your webcam feed.
4. Adjust settings such as border width, color, detection threshold, and webcam device as needed.

## License

This project is shared for learning and demonstration purposes as part of an A-Level coursework submission. See [LICENSE](./LICENSE) for details.
