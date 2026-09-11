# Image Editor

A modern desktop image editor built with **Python, OpenCV, CustomTkinter, Pillow, and YOLO**.

This project combines traditional image processing with **AI-powered object detection** in a single desktop application. It provides an interactive interface for applying image filters, transforming images, detecting objects, and managing detection results.

## Features

* Open and edit JPG, JPEG, PNG, BMP, and WebP images
* Image processing with OpenCV
* Blur, Gaussian Blur, Sharpen, Median, and Bilateral filters
* Image rotation and horizontal/vertical flipping
* Image resizing with custom dimensions
* Adjustable image noise generation
* AI-powered object detection using YOLO
* Confidence-based object detection
* Bounding boxes with detected class names and confidence scores
* Remove individual detections from the visualization
* Restore deleted detections
* Camera capture and real-time preview
* Undo / Redo history
* Export processed images as PNG or JPEG
* Modern dark-themed graphical interface
* Interactive controls and tooltips

## Object Detection

The application uses **Ultralytics YOLO** for object detection.

Detected objects are displayed with:

* Bounding boxes
* Object class
* Confidence score
* Individual object controls
* Detection summary

The detection system is integrated directly into the image editor, allowing image processing and AI-based analysis to be used together in the same workflow.

## Technologies

* Python
* OpenCV
* NumPy
* Pillow
* CustomTkinter
* Ultralytics YOLO

## Project Structure

```text
Image_editor/
│
├── Image_editor.py
├── yolov8n.pt
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mohammadmahdimoradi-git/Image-editor.git
cd Image-editor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python Image_editor.py
```

## Project Goal

The goal of this project is to combine **computer vision, image processing, and AI-based object detection** into a practical desktop application while maintaining a clean and user-friendly interface.

This project is also part of my ongoing learning and portfolio development in **Computer Vision and Artificial Intelligence**.

## Future Improvements

* More advanced image editing tools
* Additional YOLO models
* Object tracking
* Background removal
* Image segmentation
* Batch image processing
* More advanced AI-based editing features

## Author

**Mohammad Mahdi Moradi**

GitHub: [mohammadmahdimoradi-git](https://github.com/mohammadmahdimoradi-git)
