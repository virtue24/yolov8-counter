import cv2
from ultralytics import YOLO
from pathlib import Path

class ObjectInstance:
    """
    Simple class to store object state:
      - location: (x_center, y_center) in normalized coordinates (0..1)
      - id: unique integer
      - age: how many consecutive frames it has gone unmatched
      - max_age: threshold for removal
    """
    def __init__(self, location, id, max_age=5):
        self.location = location
        self.id = id
        self.age = 0
        self.max_age = max_age

    def calculate_distance(self, x, y):
        """
        Euclidean distance in normalized coordinate space.
        """
        dx = x - self.location[0]
        dy = y - self.location[1]
        return (dx*dx + dy*dy)**0.5

    def is_aged(self):
        """
        Return True if object's age has exceeded max_age.
        """
        return self.age > self.max_age

    def update_location(self, x, y, update_factor=0.75):
        """
        Update object's location.
        """
        self.location = (
            update_factor * x + (1 - update_factor) * self.location[0],
            update_factor * y + (1 - update_factor) * self.location[1]
        )

def track_objects_in_video(video_path, model_path, display_ratio=1.0, skip_frame = 3):
    # 1. Load YOLOv8 model
    model = YOLO(model_path)

    # 2. Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # List of tracked objects
    objects = []
    # We assign a unique ID to each new track
    id_counter = 0
    frame_counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # no more frames
        frame_counter += 1
        if frame_counter % skip_frame != 0:
            continue
        
        # -----------------------------
        # 3. YOLO inference on the frame
        # -----------------------------
        results = model(frame, verbose=False)
        list_of_detections = []  
        # We'll store each detection as:
        # [x1, y1, x2, y2, conf, x1n, y1n, x2n, y2n, class_name]
        #    ^ pixel coords     ^        normalized coords      ^   label
        for result in results:
            for box in result.boxes:
                # Get pixel coords
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get normalized coords
                x1n, y1n, x2n, y2n = box.xyxyn[0]

                # Confidence
                conf = float(box.conf[0])

                # Class name
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id] if model.names and cls_id in model.names else str(cls_id)

                list_of_detections.append([
                    x1, y1, x2, y2, conf, 
                    x1n, y1n, x2n, y2n, 
                    class_name
                ])

        # -----------------------------
        # 4. Match each detection to the closest object
        # -----------------------------
        matched_detection_indices = set()   # which DETECTIONS got matched
        matched_object_indices = set()      # which OBJECTS got matched

        for d_i, detection in enumerate(list_of_detections):
            x1, y1, x2, y2, conf, x1n, y1n, x2n, y2n, class_name = detection
            # Detection's center in normalized coords
            x_center = (x1n + x2n) / 2
            y_center = (y1n + y2n) / 2

            min_distance = 0.25  # threshold in normalized space
            min_obj_idx = -1

            # Find the closest existing object below this threshold
            for o_i, obj in enumerate(objects):
                # Already matched this object? skip
                if o_i in matched_object_indices:
                    continue

                dist = obj.calculate_distance(x_center, y_center)
                if dist < min_distance:
                    min_distance = dist
                    min_obj_idx = o_i

            # If we found a close-enough existing object, match them
            if min_obj_idx != -1:
                obj = objects[min_obj_idx]
                obj.update_location(x_center, y_center)
                obj.age = 0  # reset age because we matched
                matched_object_indices.add(min_obj_idx)
                matched_detection_indices.add(d_i)

        # -----------------------------
        # 5. Age unmatched objects and remove old ones
        # -----------------------------
        # (Objects not in matched_object_indices are unmatched)
        for o_i in reversed(range(len(objects))):
            if o_i not in matched_object_indices:
                objects[o_i].age += 1
                if objects[o_i].is_aged():
                    del objects[o_i]

        # -----------------------------
        # 6. Create new objects for unmatched detections
        # -----------------------------
        for d_i, detection in enumerate(list_of_detections):
            if d_i not in matched_detection_indices:
                x1, y1, x2, y2, conf, x1n, y1n, x2n, y2n, class_name = detection
                x_center = (x1n + x2n) / 2
                y_center = (y1n + y2n) / 2

                # the y value should below 0.5
                if y_center < 0.5:
                    continue

                objects.append(ObjectInstance(location=(x_center, y_center), id=id_counter))
                print(f"New object #{id_counter} at ({x_center:.2f}, {y_center:.2f})")
                id_counter += 1

        # (Optional) ------------------
        # 7. Visualization (draw boxes/tracks)
        # --------------------------------------
        # Draw bounding boxes from YOLO detections
        for detection in list_of_detections:
            x1, y1, x2, y2, conf, *rest = detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Draw object centers
        for obj in objects:
            # Convert normalized center to pixel coords
            h, w, _ = frame.shape
            cx = int(obj.location[0] * w)
            cy = int(obj.location[1] * h)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
            # Draw white rectangle as background for text
            text = f"ID:{obj.id}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            cv2.rectangle(frame, (cx, cy - 10 - text_height), (cx + text_width, cy - 10), (255, 255, 255), -1)
            # Draw text on top of the rectangle
            cv2.putText(frame, text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        # Put total object count on the frame

        # Draw white rectangle as background for total object count text
        text = f"Uretilen Borek: {id_counter-1}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        cv2.rectangle(frame, (10, 50 - text_height - 10), (10 + text_width, 50), (255, 255, 255), -1)
        # Draw total object count text on top of the rectangle
        cv2.putText(frame, text, (10, 50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        # Resize the frame for display
        frame = cv2.resize(frame, (0, 0), fx=display_ratio, fy=display_ratio)
        # Display the frame        
        cv2.imshow("Detections and Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    VIDEO_PATH = input("Enter the path to the video file: ")
    MODEL_PATH = Path(__file__).parent / 'borek_counter.pt'
    track_objects_in_video(VIDEO_PATH, MODEL_PATH, display_ratio = 0.5)
