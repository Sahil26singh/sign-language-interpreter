import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import math
import time
import os
import keyboard  # global key listener - doesn't need window focus

# --- CONFIGURATION ---
DATA_PATH = "Data2"
LABEL = input("Which letter/label are you collecting data for? (e.g. A): ").strip().upper()
if not LABEL:
    raise SystemExit("No label entered. Exiting.")
SAVE_PATH = os.path.join(DATA_PATH, LABEL)
IMG_SIZE = 300
OFFSET = 20

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Count any images already saved for this label (in case you're resuming)
counter = len([f for f in os.listdir(SAVE_PATH) if f.lower().endswith(".jpg")])
print(f"Collecting for label '{LABEL}'. {counter} image(s) already saved.")
print("Press 's' to save a frame, 'q' to quit this label.")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Used only for the HAND_CONNECTIONS index pairs (still available even with the new Tasks-API detector)
mp_hands = mp.solutions.hands

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # This installed cvzone version returns a tuple: (hands_list, img)
    hands, _ = detector.findHands(img, draw=False)
    hand_ready = False  # becomes True only if we successfully build img_white this frame

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        lmList = hand['lmList']  # list of [x, y, z] points in full-image coordinates

        # Blank white canvas the size of the full frame
        img_skeleton = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255

        # Draw bones (green lines) using the known hand connection pairs
        for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
            x1_pt, y1_pt = lmList[start_idx][0], lmList[start_idx][1]
            x2_pt, y2_pt = lmList[end_idx][0], lmList[end_idx][1]
            cv2.line(img_skeleton, (x1_pt, y1_pt), (x2_pt, y2_pt), (0, 255, 0), 3)

        # Draw joints (red dots)
        for point in lmList:
            cv2.circle(img_skeleton, (point[0], point[1]), 4, (0, 0, 255), cv2.FILLED)

        # Final 300x300 white square, centered
        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

        try:
            y1, y2 = max(0, y - OFFSET), min(img.shape[0], y + h + OFFSET)
            x1, x2 = max(0, x - OFFSET), min(img.shape[1], x + w + OFFSET)
            img_crop = img_skeleton[y1:y2, x1:x2]

            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = IMG_SIZE / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
                w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
                img_white[:, w_gap:w_cal + w_gap] = img_resize
            else:
                k = IMG_SIZE / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
                h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
                img_white[h_gap:h_cal + h_gap, :] = img_resize

            cv2.imshow("Data Preview (The Skeleton)", img_white)
            hand_ready = True
        except Exception:
            pass

    cv2.imshow("Main Webcam Feed", img)
    cv2.waitKey(1)  # still needed to refresh the OpenCV windows; return value ignored

    if keyboard.is_pressed("s"):
        if hand_ready:
            counter += 1
            cv2.imwrite(f'{SAVE_PATH}/Image_{time.time()}.jpg', img_white)
            print(f"Saved {counter} refined images to {SAVE_PATH}")
        else:
            print("No hand detected in frame yet - move your hand into view before pressing 's'.")
        time.sleep(0.3)  # debounce so one press doesn't save multiple frames

    if keyboard.is_pressed("q"):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()