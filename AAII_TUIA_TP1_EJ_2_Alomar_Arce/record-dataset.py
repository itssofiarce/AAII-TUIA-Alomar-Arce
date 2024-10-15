import os
import warnings
import cv2
import numpy as np
import mediapipe as mp

# Suprimir warnings de TensorFlow y Protobuf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Configuración e inicialización de MediaPipe y OpenCV
mp_hands = mp.solutions.hands  # Cargar la solución de manos de MediaPipe
hands = mp_hands.Hands()  # Objeto para detectar manos y sus puntos clave
mp_drawing = mp.solutions.drawing_utils  # Dibujo de landmarks y conexiones

camera = cv2.VideoCapture(0)  # Iniciar la captura de video desde la cámara predeterminada

# Contenedores para almacenar los datos del dataset
X = []  # Lista para almacenar los puntos clave (solo x e y)
Y = []  # Lista para almacenar las etiquetas correspondientes a cada gesto

# Definición de los gestos y sus respectivas etiquetas numéricas
gesture_mapping = {'piedra': 0, 'papel': 1, 'tijeras': 2}  # Asignación de números para cada gesto

# Función para imprimir instrucciones claras para el usuario
def print_instructions():
    print("\n=== Instrucciones ===")
    print("Presiona 'p' para etiquetar Piedra")
    print("Presiona 'a' para etiquetar Papel")
    print("Presiona 't' para etiquetar Tijeras")
    print("Presiona 'q' para salir\n")

# Imprimir instrucciones iniciales
print_instructions()

# Bucle principal para captura de gestos y puntos clave
while camera.isOpened():
    success, image = camera.read()  # Leer un frame de la cámara
    if not success:
        print("Ignorando frame vacío.")
        continue

    # Procesar la imagen para detectar manos y extraer puntos clave
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Capturar la tecla presionada por el usuario
    key = cv2.waitKey(1) & 0xFF

    # Si se detectan manos, dibujar los puntos clave
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer solo las coordenadas x e y de cada punto clave si se presiona una tecla
            if key == ord('p'):
                landmarks_2d = [(point.x, point.y) for point in hand_landmarks.landmark]
                X.append(landmarks_2d)
                Y.append(gesture_mapping['piedra'])
                print("Gesto Piedra capturado")
            elif key == ord('a'):
                landmarks_2d = [(point.x, point.y) for point in hand_landmarks.landmark]
                X.append(landmarks_2d)
                Y.append(gesture_mapping['papel'])
                print("Gesto Papel capturado")
            elif key == ord('t'):
                landmarks_2d = [(point.x, point.y) for point in hand_landmarks.landmark]
                X.append(landmarks_2d)
                Y.append(gesture_mapping['tijeras'])
                print("Gesto Tijeras capturado")

    # Si no se detectan manos, imprimir un mensaje de error y las instrucciones
    else:
        if key == ord('p'):
            print("\nNo hay manos en la imagen. No se puede capturar el gesto 'Piedra'.")
            print_instructions()
        elif key == ord('a'):
            print("\nNo hay manos en la imagen. No se puede capturar el gesto 'Papel'.")
            print_instructions()
        elif key == ord('t'):
            print("\nNo hay manos en la imagen. No se puede capturar el gesto 'Tijeras'.")
            print_instructions()

    # Mostrar la imagen en tiempo real
    cv2.imshow('Manos MediaPipe', image)

    # Salir del bucle si se presiona 'q'
    if key == ord('q'):
        break

# Guardar los datos capturados en archivos .npy (puntos clave y etiquetas)
#np.save('mi_dataset_X.npy', np.array(X))  
#np.save('mi_dataset_Y.npy', np.array(Y))  
#print("Datos guardados en 'mi_dataset_X.npy' y 'mi_dataset_Y.npy'.")

# Liberar recursos y cerrar las ventanas
camera.release()
cv2.destroyAllWindows()

# Imprimir los datos cargados
print("Datos cargados:")
print("X:",X)
print("Y:",Y)

