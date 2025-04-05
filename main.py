import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Definir el objeto virtual (cuadro)
class Rect:
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size
        self.drag = False

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        overlay = img.copy()

        # Cambia color si está siendo arrastrado
        color = (0, 255, 0) if self.drag else (255, 0, 0)

        # Cuadro semi-transparente
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Borde blanco
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

# Crear instancia del cuadro
rect = Rect([250, 200], [100, 100])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener coordenadas de índice y pulgar
            x1 = int(hand_landmarks.landmark[8].x * w)   # Índice
            y1 = int(hand_landmarks.landmark[8].y * h)
            x2 = int(hand_landmarks.landmark[4].x * w)   # Pulgar
            y2 = int(hand_landmarks.landmark[4].y * h)

            # Posición del cursor (dedo índice)
            cx, cy = x1, y1

            # Distancia entre índice y pulgar
            distancia = np.hypot(x2 - x1, y2 - y1)

            # Detectar si se está "agarrando"
            agarrando = distancia < 40

            # Verificar si se toca el cuadro
            rx, ry = rect.pos
            rw, rh = rect.size

            if agarrando:
                if rx < cx < rx + rw and ry < cy < ry + rh:
                    rect.drag = True
            else:
                rect.drag = False

            if rect.drag:
                rect.pos = [cx - rw // 2, cy - rh // 2]

    rect.draw(img)
    cv2.imshow("Arrastrar y Soltar Virtual", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
