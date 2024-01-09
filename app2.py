from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)
cap = None
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("modeloEigenFaceRecognizer.xml")

def create_labels_mapping(data_path):
    try:
        people_list = os.listdir(data_path)
        labels_mapping = {label: person_name for label, person_name in enumerate(people_list)}
        print('labels')
        print(labels_mapping)
        print("--------------")
        return labels_mapping
    except Exception as e:
        print(f"¡Error! create_labels_mapping: {e}")

def recognize_faces_realtime(frame, labels_mapping):
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))  # Redimensionar directamente con OpenCV

            label, confidence = face_recognizer.predict(face_roi_resized)
            person_name = labels_mapping.get(label, f'Persona {label}')

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame
    except Exception as e:
        print(f"¡Error! recognize_faces_realtime: {e}")
        return frame

def generate(labels_mapping):
    global cap
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_recognition = recognize_faces_realtime(frame, labels_mapping)

        (flag, encodedImage) = cv2.imencode(".jpg", frame_with_recognition)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

    # Liberar la cámara cuando el generador termina
    if cap is not None:
        cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    labels_mapping = create_labels_mapping("output_faces_folder")
    return Response(generate(labels_mapping), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/training")
def training():
    # Agrega la lógica que deseas ejecutar para la página de entrenamiento
    return render_template("training.html")

@app.route("/upload")
def upload():
    # Agrega la lógica que deseas ejecutar para la página de carga
    return render_template("upload.html")

@app.route("/realtime")
def realtime():
    # Agrega la lógica que deseas ejecutar para la página en tiempo real
    return render_template("realtime.html")

if __name__ == "__main__":
    app.run(debug=False)
