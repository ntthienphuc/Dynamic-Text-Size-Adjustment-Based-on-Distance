import cv2 as cv
import numpy as np
import mediapipe as mp

# Constants defining the indices of the left and right iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Constants for the known size of a reference object (e.g., face) in the scene
REFERENCE_OBJECT_WIDTH = 20  # Width of the reference object in centimeters (just an example)
REFERENCE_OBJECT_PIXELS = 200  # Width of the reference object in pixels (measure this in your scene)

# Known distance from the camera to the reference object
KNOWN_DISTANCE = 50  # in centimeters (just an example)

# Function to calculate distance from webcam to object
def calculate_distance_to_object(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to map distance to font size
def map_distance_to_font_size(distance):
    min_font_size = 20  # Minimum font size
    max_font_size = 100  # Maximum font size
    min_distance = 10  # Minimum distance
    max_distance = 2000  # Maximum distance

    # Map distance to font size within the specified range
    font_size = np.interp(distance, [min_distance, max_distance], [min_font_size, max_font_size])
    return int(font_size)

mp_face_mesh = mp.solutions.face_mesh
cap = cv.VideoCapture(0)

# Initial font parameters
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1  # Initial font scale
font_thickness = 2  # Initial font thickness

# Text to display
text_to_display = "Hello, World!"
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
  while True:
      ret, frame = cap.read()
      if not ret:
          break
      frame = cv.flip(frame, 1)
      rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
      img_h, img_w = frame.shape[:2]
      results = face_mesh.process(rgb_frame)
      if results.multi_face_landmarks:
          mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                  results.multi_face_landmarks[0].landmark])
          (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
          (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
          center_left = np.array([l_cx, l_cy], dtype=np.int32)
          center_right = np.array([r_cx, r_cy], dtype=np.int32)
          cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
          cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
          # Calculate the focal length based on known reference object size and its size in pixels
          focal_length = (REFERENCE_OBJECT_PIXELS * KNOWN_DISTANCE) / REFERENCE_OBJECT_WIDTH
          # Calculate the size of the object (iris) in pixels
          object_width_pixels = max(l_radius, r_radius) * 2
          # Estimate distance to the object (iris) using the focal length and object size in pixels
          distance = calculate_distance_to_object(REFERENCE_OBJECT_WIDTH, focal_length, object_width_pixels)
          # print("Distance to object (iris): {:.2f} cm".format(distance))

          # Map distance to font size
          font_scale = map_distance_to_font_size(distance) / 40
          # Define the text size
          text_size = cv.getTextSize(text_to_display, font, font_scale, font_thickness)[0]
          # Calculate the position to display the text
          text_x = int((img_w - text_size[0]) / 2)
          text_y = int((img_h + text_size[1]) / 2)
          # Draw the text on the frame
          cv.putText(frame, text_to_display, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness, cv.LINE_AA)

      cv.imshow('img', frame)
      key = cv.waitKey(1)
      if key == ord('q'):
          break

cap.release()
cv.destroyAllWindows()
