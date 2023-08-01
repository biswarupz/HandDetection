import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1288)
cap.set(4, 720)

drawing_color = (32, 120, 255)
drawing_thickness = 6
prev_x, prev_y = None, None

canvas = np.zeros((720, 1288, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1288, 720))
    img = cv2.flip(img, 1)  # Flip the image horizontally
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), drawing_color, drawing_thickness)

                    prev_x, prev_y = cx, cy

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    img_with_canvas = cv2.addWeighted(img, 1, canvas, 1, 0)

    cv2.imshow("Image", img_with_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
