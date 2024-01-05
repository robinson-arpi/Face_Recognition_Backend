import cv2
import os
cap = cv2.VideoCapture("16.mp4") #Seleccionamos el video
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
count = 0
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    k = cv2.waitKey(1)
    if k == 27:
        break
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('rostros/rostro_{}.jpg'.format(count),rostro)
        cv2.imshow('rostro',rostro)
        count = count + 1
    cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)

    cv2.imshow('frame',frame)
    print(count)
cap.release()
cv2.destroyAllWindows()