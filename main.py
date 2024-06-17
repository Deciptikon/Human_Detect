import torch
import cv2
import numpy as np

# Загрузка предобученной модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Константы
AVERAGE_HUMAN_HEIGHT = 1.7  # Средняя высота человека в метрах
FOCAL_LENGTH = 800  # Примерное фокусное расстояние камеры (в пикселях)

def calculate_distance(height_in_pixels, average_height=AVERAGE_HUMAN_HEIGHT, focal_length=FOCAL_LENGTH):
    return (average_height * focal_length) / height_in_pixels

def detect_person_and_calculate_distance(frame):
    results = model(frame)

    # Извлечение координат детекции человека
    person_detections = []
    for *box, conf, cls in results.xyxy[0]:
        if cls == 0:  # Класс 0 - человек
            person_detections.append(box)

    if not person_detections:
        return None, None, None

    # Берем первую детекцию
    x1, y1, x2, y2 = person_detections[0]
    height_in_pixels = y2 - y1

    # Расчет расстояния
    distance = calculate_distance(height_in_pixels)

    # Расчет смещения от центра кадра
    frame_center_x = frame.shape[1] / 2
    person_center_x = (x1 + x2) / 2
    offset_in_pixels = person_center_x - frame_center_x

    # Преобразование смещения в градусы
    angle_offset = np.degrees(np.arctan(offset_in_pixels / FOCAL_LENGTH))
    bbox = [x1, y1, x2, y2]

    return bbox, distance, angle_offset

def main():
    # Загрузка видео
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.resize(frame, (800, 600))
        
        bbox, distance, angle_offset = detect_person_and_calculate_distance(frame)
        
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Рисуем бокс вокруг человека
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if distance is not None:
            print(f"Distance to person: {distance:.2f} meters")
            print(f"Offset from center: {angle_offset:.2f} degrees")

            # Отображение результата на кадре
            cv2.putText(frame, f"Дистанция: {distance:.2f}м", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Угол: {angle_offset:.2f} градусов", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отображение кадра
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()