from ultralytics import YOLO
import cv2  # Importing OpenCV to save the image

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('runs/detect/train5/weights/last.pt')

# Perform object detection on an image using the model
results_list = model('000005.jpg')

# Assuming only one image was processed, get the first item from results_list
results = results_list[0]

# Accessing boxes from the results
boxes = results.boxes.xyxy
if boxes is not None:
    boxes = boxes.cpu().numpy()  # converting tensor to numpy array
else:
    boxes = []

# Accessing probabilities from the results
probs = results.boxes.conf
if probs is not None:
    probs = probs.cpu().numpy()  # converting tensor to numpy array
else:
    probs = []

# Getting original image as numpy array
orig_img = results.orig_img

# Iterating over boxes and drawing them on the original image
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    
    color = (0, 255, 0)  # green color for the box
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
    label = f"Confidence: {probs[i]:.2f}"
    cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Saving the original image with drawn boxes and confidence
cv2.imwrite('output.png', orig_img)
