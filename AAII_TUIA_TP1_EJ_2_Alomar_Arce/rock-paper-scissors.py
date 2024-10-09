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

            # Extraer las coordenadas x e y de los landmarks (21 puntos)
            landmarks_2d = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            landmarks_2d = np.array(landmarks_2d).flatten().reshape(1, 42)  # Aplanar los puntos a 1x42

            # Realizar la predicción
            prediction = model.predict(landmarks_2d)
            predicted_class = np.argmax(prediction)  # Obtener el índice de la clase predicha

            # Mostrar el gesto reconocido en la pantalla
            gesture = gesture_mapping[predicted_class]
            cv2.putText(frame, f'Gesto: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar la imagen en tiempo real con el gesto detectado
    cv2.imshow('Piedra, Papel o Tijeras', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
