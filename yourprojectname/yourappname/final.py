import torch
import torchvision
import cv2

# Download the detr model from the torch hub
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)


# Define a function to draw boxes around objects
def draw_boxes(frame, outputs):
    # Get the predicted bounding boxes
    boxes = outputs['pred_boxes'].detach().cpu().numpy()

    # Check if the key 'pred_labels' exists in the outputs dictionary
    if 'pred_labels' in outputs:
        labels = outputs['pred_labels'].detach().cpu().numpy()

        # Draw a box around each object
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            # Convert the box coordinates to integers
            x1, y1, x2, y2 = box.astype(int)

            # Draw the box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # If 'pred_labels' is not present, only draw bounding boxes without labels
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Start the webcam and loop over the frames
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    # Run the model on the frame
    outputs = model(frame.unsqueeze(0))

    # Draw the boxes around the objects
    draw_boxes(frame, outputs)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
