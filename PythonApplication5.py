import cv2
import mediapipe as mp
import math
import time
import os
import wave
import struct
import random
import winsound  # Thư viện âm thanh có sẵn trên Windows

# --- HÀM TỰ ĐỘNG TẠO FILE ÂM THANH "TÁCH" VÀ "TING" ---
def init_sound_effects():
    sample_rate = 44100

    # 1. Tạo tiếng "Tách" sắc gọn (15 mili-giây) khi 2 ngón tay chạm nhau
    if not os.path.exists("tap.wav"):
        n_samples = int(sample_rate * 0.015)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # Biên độ giảm nhanh tạo độ nảy cơ học ("tách")
            env = math.exp(-i / (n_samples * 0.18))
            freq = 1700 * math.exp(-i / (n_samples * 0.35)) + 450
            noise = (random.random() * 2 - 1) * 0.35 if i < 90 else 0.0
            val = (math.sin(2 * math.pi * freq * t) * 0.75 + noise) * env
            sample = int(max(-1.0, min(1.0, val)) * 32767)
            frames.extend(struct.pack('<h', sample))
        with wave.open("tap.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(frames)

    # 2. Tạo tiếng "Ting" êm tai (60 mili-giây) khi chữ cái được in ra màn hình
    if not os.path.exists("commit.wav"):
        n_samples = int(sample_rate * 0.06)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            env = math.exp(-i / (n_samples * 0.3))
            val = math.sin(2 * math.pi * 1046.5 * t) * env * 0.45  # Nốt Đô cao (C6)
            sample = int(max(-1.0, min(1.0, val)) * 32767)
            frames.extend(struct.pack('<h', sample))
        with wave.open("commit.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(frames)

def play_sound(filename):
    # SND_ASYNC giúp phát âm thanh chạy ngầm, không làm đứng hình camera
    winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)

# Khởi tạo file âm thanh ngay khi bật chương trình
init_sound_effects()

# 1. Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = {
    "THUMB": mp_hands.HandLandmark.THUMB_TIP,          # Mốc 4: Ngón cái
    "INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP,   # Mốc 8: Ngón trỏ
    "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP, # Mốc 12: Ngón giữa
    "RING": mp_hands.HandLandmark.RING_FINGER_TIP,     # Mốc 16: Ngón áp út
    "PINKY": mp_hands.HandLandmark.PINKY_TIP           # Mốc 20: Ngón út
}

# 2. BẢNG QUY TẮC CỬ CHỈ CỦA BẠN (A-Z và 0-9)
GESTURE_RULES = {
    ("R_THUMB", "R_INDEX"):  {1: "A", 2: "B", 3: "C", 4: "D"},
    ("R_THUMB", "R_MIDDLE"): {1: "E", 2: "F", 3: "G", 4: "H", 5: "J"},
    ("R_THUMB", "R_RING"):   {1: "I", 2: "K", 3: "L"},
    ("R_THUMB", "R_PINKY"):  {1: "M", 2: "N", 3: "O", 4: "P", 5: "W"},
    ("L_THUMB", "L_MIDDLE"): {1: "Q", 2: "R", 3: "S", 4: "T", 5: "Z"},
    ("L_THUMB", "L_RING"):   {1: "U", 2: "V", 3: "Y", 4: "X"},
    ("L_THUMB", "L_PINKY"):  {1: "0", 2: "1", 3: "2", 4: "3", 5: "4", 6: "5", 7: "6", 8: "7", 9: "8", 10: "9"},
    ("L_THUMB", "L_INDEX"):  {1: "<SPACE>", 2: "<BACKSPACE>", 3: "<CLEAR>"},
}

TOUCH_DIST = 35       # Khoảng cách (pixel) tính là ĐÃ CHẠM
RELEASE_DIST = 55     # Khoảng cách (pixel) tính là ĐÃ NHẢ RA
TAP_TIMEOUT = 0.55    # Thời gian chờ (giây) để chốt ký tự

pair_states = {
    pair: {"touching": False, "taps": 0, "last_time": 0.0}
    for pair in GESTURE_RULES
}

typed_text = ""
last_action_msg = ""
last_action_time = 0.0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        h, w, c = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        current_points = {}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_label = handedness.classification[0].label
                prefix = "R_" if hand_label == "Right" else "L_"

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(image, f"{hand_label} Hand", (wx - 30, wy + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                for finger_name, landmark_id in FINGER_TIPS.items():
                    lm = hand_landmarks.landmark[landmark_id]
                    px, py = int(lm.x * w), int(lm.y * h)
                    current_points[prefix + finger_name] = (px, py)

        current_time = time.time()

        # 3. KIỂM TRA CHẠM VÀ PHÁT TIẾNG "TÁCH"
        for (p1_name, p2_name), actions in GESTURE_RULES.items():
            state = pair_states[(p1_name, p2_name)]

            if p1_name in current_points and p2_name in current_points:
                x1, y1 = current_points[p1_name]
                x2, y2 = current_points[p2_name]
                dist = math.hypot(x2 - x1, y2 - y1)

                # Khoảnh khắc 2 ngón vừa chạm nhau -> Phát tiếng "Tách"
                if dist < TOUCH_DIST and not state["touching"]:
                    state["touching"] = True
                    state["taps"] += 1
                    state["last_time"] = current_time
                    play_sound("tap.wav")  # <--- PHÁT TIẾNG TÁCH TẠI ĐÂY

                elif dist > RELEASE_DIST and state["touching"]:
                    state["touching"] = False

                if state["taps"] > 0:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    max_tap = max(actions.keys())
                    preview_char = actions.get(state["taps"], actions.get(max_tap))

                    if state["touching"]:
                        cv2.circle(image, (cx, cy), 16, (0, 255, 0), cv2.FILLED)

                    cv2.putText(image, f"x{state['taps']} -> [{preview_char}]", (cx + 15, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            else:
                state["touching"] = False

            # 4. XỬ LÝ XUẤT CHỮ CÁI VÀ PHÁT TIẾNG "TING" KHI CHỐT CHỮ
            if state["taps"] > 0 and not state["touching"] and (current_time - state["last_time"]) > TAP_TIMEOUT:
                tap_count = state["taps"]
                state["taps"] = 0

                output = actions.get(tap_count)
                if output is None and tap_count > 0:
                    max_tap = max(actions.keys())
                    output = actions.get(max_tap)

                if output:
                    play_sound("commit.wav")  # <--- PHÁT TIẾNG TING NHẸ KHI CHỮ HIỆN RA
                    if output == "<SPACE>":
                        typed_text += " "
                        last_action_msg = "SPACE (Dau cach)"
                    elif output == "<BACKSPACE>":
                        typed_text = typed_text[:-1]
                        last_action_msg = "BACKSPACE (Xoa 1 ky tu)"
                    elif output == "<CLEAR>":
                        typed_text = ""
                        last_action_msg = "CLEAR (Xoa tat ca)"
                    else:
                        typed_text += output
                        last_action_msg = f"Da go: '{output}' ({tap_count} cham)"
                    
                    last_action_time = current_time

        # 5. GIAO DIỆN HIỂN THỊ VĂN BẢN (HUD)
        cv2.rectangle(image, (0, 0), (w, 85), (25, 25, 25), cv2.FILLED)
        cv2.putText(image, "VAN BAN: " + typed_text + "|", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if current_time - last_action_time < 1.5:
            cv2.putText(image, last_action_msg, (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)

        cv2.rectangle(image, (0, h - 35), (w, h), (25, 25, 25), cv2.FILLED)
        cv2.putText(image, "Tay Phai: A-P,W | Tay Trai: Q-Z | Ut Trai: So 0-9 | Tro Trai: Space/Xoa | ESC: Thoat",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)

        cv2.imshow("Ban Phim Cu Chi Cho Nguoi Khiem Ngon", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
