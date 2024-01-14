from flask import Flask, render_template, Response, request, jsonify, send_file
from moviepy.editor import VideoFileClip
import cv2
import os
import numpy as np

import time

app = Flask(__name__)
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(0)     #Variable para person_recognition

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("modeloEigenFaceRecognizer.xml")

# Carga de el modelo preentrenado de Yolov3
configuracion = "model/yolov3.cfg"
pesos_red = "model/yolov3.weights"
etiquetas = open("model/coco.names").read().split("\n")

# Colores para los rectángulos
colores = np.random.randint(0, 255, size=(len(etiquetas), 3), dtype="uint8")

# Creación del modelo de red 
red_yolo = cv2.dnn.readNetFromDarknet(configuracion, pesos_red)

def person_recognition_model():
    global video
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Dimensiones del frame
        height, width, _ = frame.shape

        desc = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        lan = red_yolo.getLayerNames()
        lan = [lan[i - 1] for i in red_yolo.getUnconnectedOutLayers()]
        red_yolo.setInput(desc)
        salidas_red = red_yolo.forward(lan)

        cajas = []
        confi = []
        class_IDs = []
        for output in salidas_red:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.7 and classID == 0:
                    box = detection[:4] * np.array([width, height, width, height])
                    #print("box:", box)
                    (x_center, y_center, w, h) = box.astype("int")
                    #print((x_center, y_center, w, h))
                    x = int(x_center - (w / 2))
                    y = int(y_center - (h / 2))
                    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cajas.append([x, y, w, h])
                    confi.append(float(confidence))
                    class_IDs.append(classID)
        if classID == 0:
            idx = cv2.dnn.NMSBoxes(cajas, confi, 0.5, 0.5)
            if len(idx) > 0:
                for i in idx:
                    (x, y) = (cajas[i][0], cajas[i][1])
                    (w, h) = (cajas[i][2], cajas[i][3])
                    color = colores[class_IDs[i]].tolist()
                    # text = "{}: {:.3f}".format(etiquetas[classIDs[i]], confidences[i])
                    text = "{}: {:.3f}".format('Persona', confi[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
    video.release()

def extract_faces_from_video(video_path, output_folder, face_cascade_path, max_captures=50):
    try:
        video_capture = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Obtener el nombre del video sin la extensión
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Crear la carpeta de salida con el nombre del video
        output_video_folder = os.path.join(output_folder, video_name)
        os.makedirs(output_video_folder, exist_ok=True)

        count = 0
        while count < max_captures:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_file_path = os.path.join(output_video_folder, f'face_{count}.jpg')
                cv2.imwrite(face_file_path, face)
                count += 1

            if count >= max_captures:
                break

        video_capture.release()
        return video_name
    except Exception as e:
        print(f"¡Error! extract_faces_from_video: {e}")

def train_face_recognizer(data_path, model_path='modeloEigenFaceRecognizer.xml'):
    try:
        people_list = os.listdir(data_path)
        
        labels = []
        faces_data = []
        label = 0

        for name_dir in people_list:
            person_path = os.path.join(data_path, name_dir)
            for file_name in os.listdir(person_path):
                labels.append(label)
                face = cv2.imread(os.path.join(person_path, file_name), 0)
                faces_data.append(face)

            label += 1

        # Redimensionar todas las imágenes al mismo tamaño (100x100 en este caso)
        faces_data_resized = resize_images(faces_data, size=(100, 100))

        # Crear el reconocedor de rostros
        face_recognizer = cv2.face.EigenFaceRecognizer_create()

        face_recognizer.train(faces_data_resized, np.array(labels))
        face_recognizer.write(model_path)
        print("Modelo almacenado en", model_path)
    except Exception as e:
        print(f"¡Error! train_face_recognizer: {e}")


def resize_images(images, size=(100, 100)):
    try:
        resized_images = [cv2.resize(img, size) for img in images]
    except Exception as e:
        print(f"¡Error! resize_images: {e}")
    return np.array(resized_images)


def create_labels_mapping(data_path):
    try:
        people_list = os.listdir(data_path)
        labels_mapping = {label: person_name for label, person_name in enumerate(people_list)}
        return labels_mapping
    except Exception as e:
        print(f"¡Error! create_labels_mapping: {e}")
        
def recognize_faces_in_video(video_path, model_path, output_folder, output_faces_folder):
    # Cargar el modelo entrenado
    try:    
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.read(model_path)

        # Inicializar el detector de caras
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Inicializar el algoritmo de seguimiento (en este caso, KCF)
        tracker = cv2.TrackerKCF_create()

        # Obtener el mapeo de etiquetas desde las carpetas presentes en output_folder
        labels_mapping = create_labels_mapping(output_faces_folder)
        #print(labels_mapping)

        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error al abrir el video.")
            return

        # Obtener la información del video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Configurar el video de salida
        output_video_path = os.path.join(output_folder, 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir a escala de grises para la detección de caras
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar caras en el frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]

                # Redimensionar la cara al mismo tamaño que las imágenes de entrenamiento
                face_roi_resized = resize_images([face_roi], size=(100, 100))[0]

                # Realizar la predicción utilizando el modelo entrenado
                label, confidence = face_recognizer.predict(face_roi_resized)

                # Obtener el nombre correspondiente a la etiqueta
                person_name = labels_mapping.get(label, f'Persona {label}')
                #print("PErsonaa: " + person_name)
                # Dibujar el rectángulo alrededor del rostro y mostrar la identificación
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Escribir el frame en el video de salida
            out.write(frame)

        # Liberar los recursos
        cap.release()
        out.release()
    except Exception as e:        
        print(f"¡Error! recognize_faces_in_video: {e}")

def recognize_faces_realtime(frame, labels_mapping):
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))  # Redimensionar directamente con OpenCV

            label, confidence = face_recognizer.predict(face_roi_resized)
            
            # Modificación: Verificar confianza           
            if confidence < 3500:  # Ajuste d eumbral de confianza
                person_name = labels_mapping.get(label, f'Persona {label}')
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                person_name = 'Desconocido'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
    cap.release()

@app.route("/")
def index():
    global cap
    global video
    cap.release()
    video.release()
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    labels_mapping = create_labels_mapping("output_faces_folder")
    return Response(generate(labels_mapping), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_person")
def video_feed_person():
    return Response(person_recognition_model(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload")
def upload():
    # Agrega la lógica que deseas ejecutar para la página de carga
    return render_template("upload.html")

@app.route("/realtime")
def realtime():
    # Agrega la lógica que deseas ejecutar para la página en tiempo real
    return render_template("realtime.html")

@app.route("/realtime_person")
def realtime_person():
    # Agrega la lógica que deseas ejecutar para la página en tiempo real
    return render_template("personRealTime.html")

def get_video_size(file_path):
    # Devuelve el tamaño del archivo de video en bytes
    return os.path.getsize(file_path)

@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        try:
            try:
                try:
                    #Webcam
                    if 'webcamVideo' in request.files:
                        video_file = request.files['webcamVideo']
                        video_name = request.form.get('videoName')
                    #Vidoe upload    
                    else:
                        video_file = request.files['videoFile']
                        video_name = video_file.filename    
                # Verificar si se ha enviado un archivo
                except Exception as e:   
                    return jsonify({'error': 'No se envió ningún archivo de video'}), 400
        
                
                # Guardar el archivo de video en una ruta temporal                    
                temp_video_path = 'training_videos\\' + video_name + '.mp4'
                video_file.save(temp_video_path)
                
                # Extracción de rostro
                new_face = extract_faces_from_video(temp_video_path, 'output_faces_folder', 'haarcascade_frontalface_default.xml', max_captures=50)
                
                # entrenamiento
                train_face_recognizer('output_faces_folder', 'modeloEigenFaceRecognizer.xml')    
                
                return render_template('training.html', message='Agregado(a): ' + new_face, success=True)
            
            except Exception as e:   
                return jsonify({'error': 'Error al subir el video de la webcam'}), 400

        except Exception as e:
            print(e)
            return render_template('training.html', message=str(e), success=False)

    return render_template('training.html')





def convertir_a_mp4(input_path, output_path):
    # Cargar el video desde el archivo de entrada
    video = VideoFileClip(input_path)

    # Guardar el video en formato MP4 en el archivo de salida
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        
        # Verificar si se ha enviado un archivo
        if 'videoFile' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo de video'}), 400

        video_file = request.files['videoFile']

        # Verificar si el archivo tiene una extensión válida
        if video_file.filename == '' or not video_file.filename.endswith('.mp4'):
            return jsonify({'error': 'El archivo no es un video válido (.mp4)'}), 400

        # Guardar el archivo de video en una ruta temporal
        temp_video_path = 'temp_video.mp4'
        video_file.save(temp_video_path)

        # Llamar a la función para reconocer caras en el video
        recognize_faces_in_video(temp_video_path, 'modeloEigenFaceRecognizer.xml', './static/output_folder', 'output_faces_folder')
        video_input = './static/output_folder/output_video.mp4'
        video_output = './static/output_folder/output_video_conver.mp4'
         # Devolver la URL del video procesado al front-end
        convertir_a_mp4(video_input,video_output)
        video_url = './static/output_folder/output_video_conver.mp4'
        random_parameter = int(time.time())
        return render_template('upload.html', video_url=video_url, random_parameter=random_parameter)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
