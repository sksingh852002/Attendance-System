# Face Recognition-Based Attendance System

## Overview
This project utilizes the `face_recognition` library alongside `OpenCV` to create a real-time attendance system. It captures live video input, detects and recognizes known faces, and logs attendance data into a CSV file with timestamps.

## Dependencies
- `face_recognition`: For face detection and encoding.
- `cv2` (OpenCV): For video capture and frame processing.
- `numpy`: For numerical operations.
- `csv`: To log attendance records.
- `datetime`: To record timestamps.

## Workflow
1. **Video Capture:**
   - Initializes video capture from the default camera.

2. **Load Known Faces:**
   - Loads images from the `faces` directory.
   - Encodes faces for recognition.

3. **Attendance Tracking:**
   - Creates or opens a CSV file with the current date as the filename.

4. **Real-Time Recognition:**
   - Reads frames continuously.
   - Resizes and converts frames to RGB for processing.
   - Locates faces and compares them to known encodings.

5. **Logging:**
   - Writes recognized names with timestamps into the CSV file.

6. **Display:**
   - Annotates the video feed with recognized names.

7. **Exit Condition:**
   - Stops when the user presses 'q'.

## Code Breakdown

### 1. **Imports:**
```python
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
```

### 2. **Initialization:**
```python
video_capture = cv2.VideoCapture(0)
```

### 3. **Load Known Faces:**
```python
sks_image = face_recognition.load_image_file("faces/sks.jpg")
sks_encoding = face_recognition.face_encodings(sks_image)[0]

shr_image = face_recognition.load_image_file("faces/shr.jpg")
shr_encoding = face_recognition.face_encodings(shr_image)[0]

srv_image = face_recognition.load_image_file("faces/srv.jpg")
srv_encoding = face_recognition.face_encodings(shr_image)[0]  # Bug: Incorrect image used

known_face_encodings = [sks_encoding, shr_encoding]
known_face_names = ["sks", "shr", "srv"]
students = known_face_names.copy()
```

### 4. **Create CSV File:**
```python
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
csv_writer = csv.writer(f)
```

### 5. **Main Loop:**
```python
while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
```

### 6. **Face Matching and Logging:**
```python
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            csv_writer.writerow([name, current_date, datetime.now().strftime("%H:%M:%S")])

            # Display the result
            cv2.putText(frame, name + " Present", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, 2)

            if name in students:
                students.remove(name)
```

### 7. **Exit Condition:**
```python
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
```

## Issues and Recommendations
1. **Bug in Loading Encodings:**
   - `srv_encoding` incorrectly uses `shr_image` instead of `srv_image`.
2. **Efficiency:**
   - Processing every frame is computationally intensive. Consider skipping frames or using threading.
3. **File Handling:**
   - Ensure the CSV file is closed properly even if an error occurs (use `with open` statement).

## Conclusion
This code demonstrates a functional system for real-time face recognition and attendance logging. With minor bug fixes and optimizations, it can be adapted for broader applications.

