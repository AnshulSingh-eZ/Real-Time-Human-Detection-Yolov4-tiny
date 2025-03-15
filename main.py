
##--HOG slow--##

import cv2
path = r'D:\SR_Works\2nd_Task\Q5\people-detection.mp4'
cap = cv2.VideoCapture(path)
while(cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (humans, _) = hog.detectMultiScale(image , winStride=(10,10), padding=(8,8), scale=1.1)
    print("Humans : ", len(humans))
    for(x,y,w,h) in humans:
        pad_w, pad_h = int(0.15*w), int(0.15*h)
        cv2.rectangle(image, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h) , (0,255,0), 4)
    cv2.imshow("img", image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()