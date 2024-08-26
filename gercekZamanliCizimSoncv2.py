import cv2
import mediapipe as mp
import numpy as np
import random

# MediaPipe ve OpenCV nesneleri oluşturma
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# El tespiti modeli ve çizim aracını başlatma
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0)

# Çözünürlük ayarlarını değiştirme
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Webcam görüntüsünden yüksekliği ve genişliği almak için bir örnek kare okuma
ret, frame = cap.read()
if not ret:
    raise Exception("Kamera açılmadı")

height, width, _ = frame.shape

# Boş bir beyaz canvas oluşturma
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

# Başlangıçta renk
color = (0, 0, 255)  # Kırmızı

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü ters çevirme (ağızdan)
    frame = cv2.flip(frame, 1)

    # Görüntüyü RGB formatına dönüştürme
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Elin her bir noktasını çizme
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_utils, drawing_utils)

            # İşaret parmağı ve başparmak uçlarının koordinatlarını alma
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            index_finger_x = int(index_finger_tip.x * width)
            index_finger_y = int(index_finger_tip.y * height)
            thumb_x = int(thumb_tip.x * width)
            thumb_y = int(thumb_tip.y * height)

            # Parmaklar arasındaki mesafeyi hesaplama
            distance = np.sqrt((index_finger_x - thumb_x) ** 2 + (index_finger_y - thumb_y) ** 2)

            # Yumruk yapıldığını algılamak için mesafe eşiği
            fist_threshold = 30
            
            if distance < fist_threshold:
                #print("Yumruk algılandı! Renk değiştiriliyor...")
                color = (
                    random.randint(0, 255),  # B
                    random.randint(0, 255),  # G
                    random.randint(0, 255)   # R
                )

            # Çizim yapmak için mesafe eşiği
            drawing_threshold = 50
            drawing_active = distance > drawing_threshold

            # Çizim aktifse, canvas üzerine çizim yap
            if drawing_active:
                cv2.circle(canvas, (index_finger_x, index_finger_y), 7, color, -1)  # Çizim rengi ve boyutu

    # Ekranda webcam görüntüsünü ve çizimleri gösterme
    cv2.imshow('Webcam Görüntüsü', frame)
    cv2.imshow('Çizim Alanı', canvas)  # Çizim alanını beyaz arka planla gösterir

    # 'q' tuşuna basarak çıkma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()