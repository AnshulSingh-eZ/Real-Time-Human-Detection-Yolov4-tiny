import cv2
import os
import numpy as np


DARKNET_PATH = r"D:\darknet-master"
CONFIG_PATH = os.path.join(DARKNET_PATH, "cfg", "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(DARKNET_PATH, "yolov4-tiny.weights")  
DATA_PATH = os.path.join(DARKNET_PATH, "cfg", "coco.data")
NAMES_PATH = os.path.join(DARKNET_PATH, "data", "coco.names")
VIDEO_INPUT = r"D:\SR_Works\2nd_Task\Q5\people-detection2.mp4"  
OUTPUT_VIDEO = r"D:\SR_Works\2nd_Task\Q5\output.avi" 

net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


classes = open(NAMES_PATH).read().strip().split("\n")

cap = cv2.VideoCapture(VIDEO_INPUT)
# cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes, confidences = [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]  
            class_id = np.argmax(scores) 
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))

                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(frame, f"Person {confidence:.2f}", (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
    count = 0
    if(len(indices) > 0):
        count = len(indices)
        for i in indices.flatten():
            x,y,w,h = boxes[i]
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Count {count} Humans", (25,25), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0,0,255), 2)    
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
