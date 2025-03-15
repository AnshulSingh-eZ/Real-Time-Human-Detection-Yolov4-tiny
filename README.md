This is a Realtime Human Detection program which uses yolov4-tiny for predictions, 
I designed one without yolo as well in main.py, it uses hog detection but it is too slow for predictions in a video so there was a need to scan a frame only once.
For inputing video via webcam, you can alter the input path as 
cap = cv2.VideoCapture(0)
and delete the path written, else you are good to go!!
