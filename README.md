Face Recognition Model
This project implements a face recognition system in Python using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm.

Features
Capture images for training the model

Train the LBPH face recognizer with collected images

Real-time face recognition through webcam input

Clear existing training data and retrain the model as needed

Requirements
Python 3.x

OpenCV (opencv-python)

NumPy

Usage
Capture face images using the capture functionality.

Train the model on the captured dataset.

Run the recognition module to identify faces in real-time.

Optionally clear the dataset and retrain the model to update recognition.

Setup and Installation
Install dependencies:

nginx
Copy
Edit
pip install opencv-python numpy
Run the script to capture images.

Train the recognizer.

Use the recognition script to test with webcam input.

Project Structure
capture.py — Capture face images for training

train.py — Train the LBPH model

recognize.py — Recognize faces via webcam

dataset/ — Directory storing captured images

model.yml — Trained face recognition model file

Notes
Ensure proper lighting and positioning during image capture for better accuracy.

The system works best with a consistent background and frontal faces.

