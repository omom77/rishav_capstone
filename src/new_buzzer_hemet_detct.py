from ultralytics import YOLO
import cv2
import math
import RPi.GPIO as GPIO
import time

# GPIO setup
LED_PIN = 17  # adjust as per your wiring
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Load model
model = YOLO("../weights/hemletYoloV8_100epochs.pt")
print(model.names)  # should show {0: 'head', 1: 'helmet', 2: 'person'}

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = ["head", "helmet", "person"]

def trigger_alert():
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(LED_PIN, GPIO.LOW)

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Failed to grab frame")
            trigger_alert()
            continue

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            helmet_detected = False
            head_detected = False

            for box in boxes:
                cls = int(box.cls[0])
                label = classNames[cls]

                if label == "helmet":
                    helmet_detected = True
                elif label == "head":
                    head_detected = True

            # Helmet violation logic
            if head_detected and not helmet_detected:
                print("⚠️ Helmet NOT detected — triggering LED!")
                trigger_alert()

        # Optional: Show live video (remove if running headless)
        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Manually interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Resources released")
