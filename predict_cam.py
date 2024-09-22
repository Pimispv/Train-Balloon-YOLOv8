import os
import cv2
from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.4


# Video capture setup (replace 0 for external camera)
cap = cv2.VideoCapture(0)


while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from video stream.")
        break

    # Perform object detection on the captured frame
    results = model(frame)[0]

    # Draw bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
           cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
           cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


    # Display the frame with detected balloons (optional)
    cv2.imshow('Balloon Detection', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
