import cv2
from ultralytics import YOLO

def detect_video_objects(video_path, model_path, display_ratio):
    """
    Detect objects in a video using a YOLOv8 model.

    :param video_path: path to the input video
    :param model_path: path to the trained YOLOv8 weights (e.g., 'best.pt')
    """

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open a connection to the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Read and process frames in a loop
    while True:
        ret, frame = cap.read()

        if not ret:
            # If the frame is not grabbed successfully, we're done
            break

        # Run the YOLO model on the frame
        results = model(frame)

        # `results` is a list of `ultralytics.yolo.engine.results.Results` objects
        # Usually, for a single image, it's a list with just 1 Results object: results[0]
        for result in results:
            # Each `result.boxes` is a Boxes object with info about each detected box
            for box in result.boxes:
                # Get bounding box coordinates in [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence
                conf = float(box.conf[0])

                # Get the predicted class index and name
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id] if model.names and cls_id in model.names else str(cls_id)

                # Draw the bounding box
                color = (0, 255, 0)  # BGR format
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Put the label text above the box
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),  # a little above the box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # Resize the frame to a smaller size for faster processing
        frame = cv2.resize(frame, (0, 0), fx=display_ratio, fy=display_ratio)
       
        # Display the frame with drawn bounding boxes
        cv2.imshow("YOLOv8 Detection", frame)

        # Exit loop if 'Esc' is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the Esc key
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace with your paths
    video_path = input("Enter the path to the video file: ")
    model_path = input("Enter the path to the trained weight: ")

    detect_video_objects(video_path, model_path, display_ratio = 0.5)
