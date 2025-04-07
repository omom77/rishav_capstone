from ultralytics import YOLO
import cv2
import math
import RPi.GPIO as GPIO
import time

# Initialize GPIO
LED_PIN = 18  # Using GPIO18 (physical pin 12)
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Ensure LED starts off

# Camera settings
width = 640 
height = 480

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Load YOLO model
model = YOLO("../weights/hemletYoloV8_100epochs.pt")
print(model.names) 

# Object classes
classNames = ["head", "helmet", "person"]

# Variables for LED control
last_helmet_time = 0
led_duration = 1  # LED stays on for 1 second

try:
    while True:
        success, img = cap.read()
        if not success:
            break
            
        results = model(img, stream=True)
        helmet_detected = False
        
        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
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
                
                # Check if helmet is detected
                if classNames[cls] == "helmet" and confidence > 0.5:
                    helmet_detected = True
                
                # Display class name
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, classNames[cls], org, font, 1, (255, 0, 0), 2)
        
        # Control LED based on helmet detection
        current_time = time.time()
        if helmet_detected:
            GPIO.output(LED_PIN, GPIO.HIGH)
            last_helmet_time = current_time
            print("Helmet detected - LED ON")
        elif current_time - last_helmet_time > led_duration:
            GPIO.output(LED_PIN, GPIO.LOW)
        
        # Display video feed
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Resources released")
