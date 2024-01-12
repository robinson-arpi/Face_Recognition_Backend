# Función del código del archivo auxiliar

import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

def person_recognition_HOG():
      # Inicializar el detector de personas HOG
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Iniciar la captura de video de la webcam
    cap = cv2.VideoCapture(0)
    scale = 1.0

    while True:
        # Leer frame por frame
        ret, frame = cap.read()

        #resize image -scale
        w = int(frame.shape[1] / scale)
        frame = imutils.resize(frame, width=w)

        # Detectar personas en el frame
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),  padding=(16, 16), scale=1.05)

        # Dibujar rectángulos alrededor de las personas
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Aplicación del supresión no máxima
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # Mostrar el frame resultante
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        
def person_recognition_TRACK():
    video = cv2.VideoCapture(0)
    i=0

    while True:
        ret, frame = video.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i == 20:
            bgGray = gray
        if i > 20:
            dif = cv2.absdiff(gray, bgGray)
            _, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
            cnts,_= cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                if area > 9000:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        i = i+1