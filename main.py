import torch
import cv2
import numpy as np

# Загрузка предобученной модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Константы
AVERAGE_HUMAN_HEIGHT = 1.7  # Средняя высота человека в метрах
FOCAL_LENGTH = 800  # Примерное фокусное расстояние камеры (в пикселях)
SEARCH_RADIUS = 50

def calculate_distance(height_in_pixels, average_height=AVERAGE_HUMAN_HEIGHT, focal_length=FOCAL_LENGTH):
    return (average_height * focal_length) / height_in_pixels

def extract_search_area(frame, bbox, radius=SEARCH_RADIUS):
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]

    # Определяем новую область поиска с учетом радиуса
    new_x1 = max(0, int(x1) - radius)
    new_y1 = max(0, int(y1) - radius)
    new_x2 = min(width, int(x2) + radius)
    new_y2 = min(height, int(y2) + radius)

    search_area = frame[new_y1:new_y2, new_x1:new_x2]
    return search_area, (new_x1, new_y1, new_x2, new_y2)

def summ_bbox(bbox_out, bbox_in):
    print(bbox_out)
    print(bbox_in)
    return [bbox_out[0] + bbox_in[0],
            bbox_out[1] + bbox_in[1],
            bbox_out[0] + bbox_in[2],
            bbox_out[1] + bbox_in[3]]

def detect_person(frame):
    results = model(frame)

    # Извлечение координат детекции человека
    person_detections = []
    for *box, conf, cls in results.xyxy[0]:
        if cls == 0:  # Класс 0 - человек
            person_detections.append(box)

    if not person_detections:
        return None
    
    x1, y1, x2, y2 = person_detections[0]
    bbox = (x1, y1, x2, y2)

    return bbox

def calculate_parameters(frame, bbox):
    # Берем первую детекцию
    x1, y1, x2, y2 = bbox
    height_in_pixels = y2 - y1

    # Расчет расстояния
    distance = calculate_distance(height_in_pixels)

    # Расчет смещения от центра кадра
    frame_center_x = frame.shape[1] / 2
    person_center_x = (x1 + x2) / 2
    offset_in_pixels = person_center_x - frame_center_x

    # Преобразование смещения в градусы
    angle_offset = np.degrees(np.arctan(offset_in_pixels / FOCAL_LENGTH))
    
    return distance, angle_offset

def main():
    # Загрузка видео
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        return
    
    
    bbox = [0, 0, 9999, 9999]
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.resize(frame, (800, 600))
        
        rez_frame, around_bbox = extract_search_area(frame, bbox, radius=SEARCH_RADIUS)
        
        min_bbox = detect_person(rez_frame)
        if min_bbox is not None:
            bbox = summ_bbox(around_bbox, min_bbox)
            distance, angle_offset = calculate_parameters(frame, bbox)

            x1, y1, x2, y2 = map(int, bbox)
            # Рисуем бокс вокруг человека
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Рисуем поисковой бокс
            x1, y1, x2, y2 = map(int, around_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            print(f"Distance to person: {distance:.2f} meters")
            print(f"Offset from center: {angle_offset:.2f} degrees")

            # Отображение результата на кадре
            cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Offset: {angle_offset:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отображение кадра
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()