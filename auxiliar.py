# Reconocimiento de personas con HOG
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import time

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)
t1=0

while True:
        ret, frame = cap.read()
        
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
        image = imutils.resize(frame, width=min(400, frame.shape[1]))

	# detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        t2=t1
        t1 = time.time()
        fps = 1/(t1 - t2)
        cv2.putText(image, 'FPS : {:.2f}'.format(fps), (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        
	# show the output image
        cv2.imshow("OpenCV", image)
        if (cv2.waitKey(1) == ord('s')):
             break

cap.release()
cv2.destroyAllWindows()