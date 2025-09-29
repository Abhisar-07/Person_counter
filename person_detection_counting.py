import cv2
import torch

# --- CONFIGURATION ---
# Set the threshold for the number of people.
# The alert will be triggered if the count exceeds this number.
PERSON_THRESHOLD = 1

# --- MODEL LOADING ---
# Load the pre-trained YOLOv5s model from PyTorch Hub
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Set model parameters
model.conf = 0.45  # Confidence threshold
model.iou = 0.50   # IoU threshold

# --- VIDEO CAPTURE ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Starting webcam feed for Crowd Management... Press 'q' to exit.")

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model(frame)

    # Initialize person count for this frame
    person_count = 0
    
    # Process detections from the pandas DataFrame
    detections_df = results.pandas().xyxy[0]
    
    # Filter detections to only include the 'person' class
    person_detections = detections_df[detections_df['name'] == 'person']
    
    # Update the person count
    person_count = len(person_detections)

    # --- DRAWING AND DISPLAY ---
    # Draw bounding boxes for each detected person
    for _, person in person_detections.iterrows():
        xmin, ymin, xmax, ymax = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
        # Draw a blue bounding box around the person
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # Display the current person count in green text
    count_text = f"Person Count: {person_count}"
    cv2.putText(frame, count_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if the person count exceeds the threshold
    if person_count > PERSON_THRESHOLD:
        # If it does, display a red alert message
        alert_text = "ALERT: Crowd Limit Exceeded!"
        cv2.putText(frame, alert_text, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        
    # Show the processed frame in a window
    cv2.imshow('Crowd Management System', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")