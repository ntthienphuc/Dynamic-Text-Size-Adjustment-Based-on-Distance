# Dynamic Font Scaling Based on Distance

This project demonstrates a method for dynamically adjusting the font size of text displayed in a video feed, based on the distance between the camera and an object (e.g., a face). It utilizes OpenCV for video processing and mediaPipe for facial landmark detection to estimate the distance to the user's face and adjust the font size of the displayed text accordingly.

## Features

- Uses OpenCV to capture and process video frames.
- Employs MediaPipe Face Mesh to detect facial landmarks and calculate the distance to the face.
- Dynamically adjusts the font size of a text message displayed on the video feed based on the estimated distance.

## Prerequisites

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

First, ensure that you have Python installed on your system. Then, install the required libraries using pip:

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
```

## Usage

To run the script, simply execute the Python file in your terminal:

```bash
python dynamic_font_scaling.py
```

Press 'q' to quit the application.

## How It Works

1. The script captures video frames from the webcam.
2. MediaPipe Face Mesh is used to detect facial landmarks in the video frame.
3. The script calculates the distance to the face by estimating the size of the iris.
4. Based on the calculated distance, the font size of the displayed text is adjusted dynamically.

## License

This project is open-sourced under the MIT license.

