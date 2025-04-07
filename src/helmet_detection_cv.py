from ultralytics import YOLO
import cv2
import math
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Initialize Pi Camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow camera to warm up
time.sleep(0.1)

# Model
model = YOLO("../weights/hemletYoloV8_100epochs.pt")
print(model.names)

# Object classes
classNames = ["head", "helmet", "person"]

try:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = frame.array
        
        # Process with YOLO
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Display text
                cv2.putText(img, f"{classNames[cls]} {confidence:.2f}",
                          (x1, max(y1-10, 20)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Pi Camera', img)
        
        # Clear the stream for next frame
        rawCapture.truncate(0)
        
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Cleanup
    camera.close()
    cv2.destroyAllWindows()
