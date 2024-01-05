import cv2
import os
import numpy as np

def extract_faces_from_video(video_path, output_folder, face_cascade_path, max_captures=50):
    print("video: " + video_path)
    video_capture = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Obtener el nombre del video sin la extensión
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Crear la carpeta de salida con el nombre del video
    output_video_folder = os.path.join(output_folder, video_name)
    os.makedirs(output_video_folder, exist_ok=True)

    count = 0
    while count < max_captures:
        print(count)
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
    print("saleee-----------------")


def resize_images(images, size=(100, 100)):
    resized_images = [cv2.resize(img, size) for img in images]
    return np.array(resized_images)

def train_face_recognizer(data_path, model_path='modeloEigenFaceRecognizer.xml'):
    people_list = os.listdir(data_path)
    print('Lista de personas: ', people_list)

    labels = []
    faces_data = []
    label = 0

    for name_dir in people_list:
        person_path = os.path.join(data_path, name_dir)
        print('Leyendo las imágenes')

        for file_name in os.listdir(person_path):
            print('Rostros: ', name_dir + '/' + file_name)
            labels.append(label)
            face = cv2.imread(os.path.join(person_path, file_name), 0)
            faces_data.append(face)

        label += 1

    # Redimensionar todas las imágenes al mismo tamaño (100x100 en este caso)
    faces_data_resized = resize_images(faces_data, size=(100, 100))

    # Crear el reconocedor de rostros
    face_recognizer = cv2.face.EigenFaceRecognizer_create()

    print("Entrenando...")
    face_recognizer.train(faces_data_resized, np.array(labels))
    face_recognizer.write(model_path)
    print("Modelo almacenado en", model_path)



def create_labels_mapping(data_path):
    people_list = os.listdir(data_path)
    labels_mapping = {label: person_name for label, person_name in enumerate(people_list)}
    print('labels')
    print(labels_mapping)
    print("--------------")
    return labels_mapping

def recognize_faces_in_video(video_path, model_path, output_folder, output_faces_folder):
    # Cargar el modelo entrenado
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read(model_path)

    # Inicializar el detector de caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializar el algoritmo de seguimiento (en este caso, KCF)
    tracker = cv2.TrackerKCF_create()

    # Obtener el mapeo de etiquetas desde las carpetas presentes en output_folder
    labels_mapping = create_labels_mapping(output_faces_folder)
    print(labels_mapping)

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
    cv2.destroyAllWindows()

# Uso de la función
recognize_faces_in_video('test1.mp4', 'modeloEigenFaceRecognizer.xml', 'output_folder', 'output_faces_folder')


# Uso de la función con un máximo de 30 capturas
#extract_faces_from_video('Robinson Arpi.mp4', 'output_faces_folder', 'haarcascade_frontalface_default.xml', max_captures=50)

# Uso de la función
#train_face_recognizer('output_faces_folder', 'modeloEigenFaceRecognizer.xml')

