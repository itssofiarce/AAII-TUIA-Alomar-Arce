import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Cargar el modelo desde el archivo .h5
model = tf.keras.models.load_model('mi_modelo.h5')

# Mostrar un resumen del modelo para verificar que se ha cargado correctamente
model.summary()
print("Modelo cargado desde 'mi_modelo.h5'.")

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurar la detección de manos
hands = mp_hands.Hands(max_num_hands=1)  # Detectar hasta 1 mano
cap = cv2.VideoCapture(0)  # Capturar desde la cámara web

# Mapeo de gestos: índices que corresponden a las etiquetas del modelo
gesture_mapping = {0: 'Piedra', 1: 'Papel', 2: 'Tijeras'}

frame_count = 0  # Contador de frames para optimizar predicción
prediction_threshold = 0.80  # Umbral de confianza del 80%
frames_to_skip = 10  # Cantidad de frames a procesar antes de predecir nuevamente

# Variables para almacenar la última predicción y su confianza
last_gesture = "Gesto no claro"
last_confidence = 0
display_duration = 10  # Duración en frames para mantener la predicción visible

while cap.isOpened():
    ret, frame = cap.read()  # Leer el frame de la cámara
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar los puntos clave de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Solo realizar la predicción cada n frames
            if frame_count % frames_to_skip == 0:
                # Extraer las coordenadas x e y de los landmarks (21 puntos)
                landmarks_2d = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                landmarks_2d = np.array(landmarks_2d).flatten().reshape(1, 42)  # Aplanar los puntos a 1x42

                # Realizar la predicción
                prediction = model.predict(landmarks_2d)
                predicted_class = np.argmax(prediction)  # Obtener el índice de la clase predicha
                confidence = prediction[0][predicted_class]  # Obtener la confianza de la predicción

                # Verificar si la confianza es mayor al umbral
                if confidence >= prediction_threshold:
                    last_gesture = f'{gesture_mapping[predicted_class]} (Confianza: {confidence:.2f})'
                else:
                    last_gesture = "Gesto no claro"
                
                last_confidence = confidence  # Almacenar la confianza de la predicción
                display_duration = 10  # Resetear la duración para mostrar el gesto

    # Mostrar el último gesto predicho por un período de tiempo
    if display_duration > 0:
        cv2.putText(frame, last_gesture, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Gesto no claro" not in last_gesture else (0, 0, 255), 
                    2, cv2.LINE_AA)
        display_duration -= 1  # Reducir la duración en cada frame

    # Mostrar la imagen en tiempo real con el gesto detectado
    cv2.imshow('Piedra, Papel o Tijeras', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Incrementar el contador de frames

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
