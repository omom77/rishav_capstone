from ultralytics import YOLO
import cv2
import math
from picamera import Picamera  # New import for Pi Camera

# Initialize Pi Camera
picam2 = Picamera()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Model
model = YOLO("../weights/hemletYoloV8_100epochs.pt")
print(model.names)

# Object classes
classNames = ["head", "helmet", "person"]

while True:
    try:
        # Capture frame from Pi Camera
        img = picam2.capture_array()
        
        # Convert from BGR to RGB (Pi Camera gives BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Verify image was captured
        if img is None or img.size == 0:
            print("Failed to capture image")
            continue
            
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
        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
