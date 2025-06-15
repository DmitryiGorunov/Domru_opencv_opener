import face_recognition
import imutils
import pickle
import time
import cv2
import os
import requests
import logging
from datetime import datetime
import threading
import socket

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = " "
TELEGRAM_CHAT_ID = " "
try:
    with open('face_enc', "rb") as f:
        data = pickle.load(f)
    logging.info("Face encodings loaded successfully")
except Exception as e:
    logging.error(f"Error loading face encodings: {str(e)}")
    data = {"encodings": [], "names": []}

RESIZE_WIDTH = 800
PROCESS_INTERVAL = 0.5  #(2 FPS)
DOOR_TIMEOUT = 30
DISPLAY_DURATION = 5

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

last_open_time = 0
last_process_time = 0
door_opened_frame = None
show_until = 0
door_open_cooldown = 0

def is_network_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def open_door():
    global door_open_cooldown
    try:

        result = requests.post('http://192.168.**.***:20880/api/open/************/********', timeout=2.0)
        if result.status_code == 200:
            logging.warning(f"Door opened in {result.elapsed.total_seconds()} sec")
            door_open_cooldown = time.time() + DOOR_TIMEOUT  # Устанавливаем таймаут
            return True
        else:
            logging.error(f"Door open failed: HTTP {result.status_code}")
    except Exception as e:
        logging.error(f"Door open error: {str(e)}")
    return False

def send_telegram_photo(image_path):

    def send():
        try:

            if not is_network_available():
                logging.warning("Network unavailable, skipping Telegram send")
                return

            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"


            with open(image_path, 'rb') as photo_file:
                photo_bytes = photo_file.read()

            files = {'photo': ('image.jpg', photo_bytes)}
            data = {'chat_id': TELEGRAM_CHAT_ID}
            response = requests.post(url, files=files, data=data, timeout=10)

            if response.status_code == 200:
                logging.info("Photo sent to Telegram")
            else:
                logging.error(f"Telegram send failed: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Telegram error: {str(e)}")


    threading.Thread(target=send, daemon=True).start()

def save_frame(frame, names, face_locations):

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")


        annotated_frame = frame.copy()


        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(annotated_frame, name, (left, top - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


        cv2.imwrite(filename, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR),
                   [cv2.IMWRITE_JPEG_QUALITY, 70])
        logging.info(f"Saved frame: {filename}")


        send_telegram_photo(filename)
    except Exception as e:
        logging.error(f"Error saving frame: {str(e)}")

print("Streaming started")
try:
    video_capture = cv2.VideoCapture('http://192.168.***.***:20880/api/camera/stream/************/********')
    if not video_capture.isOpened():
        raise ConnectionError("Failed to open video stream")
except Exception as e:
    logging.error(f"Video capture error: {str(e)}")
    exit(1)


last_annotated_frame = None
frame_count = 0

while True:

    ret, frame = video_capture.read()
    if not ret:
        logging.error("Failed to grab frame")

        video_capture.release()
        time.sleep(2)
        try:
            video_capture = cv2.VideoCapture('http://192.168.***.***:20880/api/camera/stream/************/********')
            if not video_capture.isOpened():
                logging.error("Reconnection failed")
            else:
                logging.info("Reconnected to video stream")
            continue
        except:
            logging.error("Critical error, exiting")
            break

    current_time = time.time()
    cooldown_active = current_time < door_open_cooldown


    display_frame = frame.copy()


    if current_time - last_process_time > PROCESS_INTERVAL:
        last_process_time = current_time

        try:

            small_frame = imutils.resize(frame, width=RESIZE_WIDTH)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            names = []
            casa_detected = False


            for encoding in encodings:
                try:

                    matches = face_recognition.compare_faces(data["encodings"], encoding)
                    name = "Unknown"

                    if True in matches:
                        matched_idxs = [i for i, match in enumerate(matches) if match]
                        counts = {}
                        for idx in matched_idxs:
                            name = data["names"][idx]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)

                    names.append(name)
                    if name == "casa":
                        casa_detected = True
                except Exception as e:
                    logging.error(f"Face processing error: {str(e)}")
                    names.append("Error")


            if names:
                save_frame(rgb_frame, names, face_locations)


            annotated_small_frame = small_frame.copy()
            for (top, right, bottom, left), name in zip(face_locations, names):
                cv2.rectangle(annotated_small_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(annotated_small_frame, name, (left, top - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


            last_annotated_frame = imutils.resize(annotated_small_frame, width=frame.shape[1])


            if casa_detected and not cooldown_active:
                if open_door():

                    door_opened_frame = last_annotated_frame.copy()
                    show_until = current_time + DISPLAY_DURATION
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")


    try:
        if last_annotated_frame is not None:

            cv2.imshow("Live Stream", last_annotated_frame)
        else:

            cv2.imshow("Live Stream", display_frame)
    except Exception as e:
        logging.error(f"Display error: {str(e)}")


    try:
        if door_opened_frame is not None and current_time < show_until:
            cv2.imshow("Door Opened", door_opened_frame)
        elif door_opened_frame is not None:
            cv2.destroyWindow("Door Opened")
            door_opened_frame = None
    except Exception as e:
        logging.error(f"Door display error: {str(e)}")


    if cooldown_active:
        remaining = int(door_open_cooldown - current_time)
        if frame_count % 20 == 0:
            logging.info(f"Door cooldown active: {remaining} seconds remaining")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()
