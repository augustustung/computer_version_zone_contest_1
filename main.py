import cv2
from tool.detector import ObjectDetection
from tool.tracker import CentroidTracker

obj_detection = ObjectDetection()
car_tracker = CentroidTracker()

# Initialize Model
net = cv2.dnn.readNet("./resource/yolov4-custom_best.weights", "./resource/yolov4-custom.cfg")
obj_detection.initialize_model(net)

cap = cv2.VideoCapture('resource/DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4')

while cap.isOpened():
    _, current_frame = cap.read()

    # detecting
    classes, scores, boxes = obj_detection.detect(current_frame)

    # Count and track cars that are only in the region of interest!
    objects_bbs_ids = car_tracker.region_of_interest(current_frame, boxes)

    # Draw result
    current_frame = obj_detection.draw_objects(current_frame, classes, scores, objects_bbs_ids)

    cv2.imshow("Image", current_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()