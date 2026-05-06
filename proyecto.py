import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

colores = {
    "car": (255, 0, 255),
    "bicycle": (255, 0, 0),
    "traffic light": (0, 165, 255),
    "person": (0, 0, 255)
}

objetos_disponibles = ["car", "bicycle", "traffic light", "person"]

print("1. Detectar todos")
print("2. Detectar solo uno")
print("3. Detectar Varios")

opcion = input("Seleccionar una opción: ")

if opcion == "1":
    objetos_seleccionados = objetos_disponibles
elif opcion == "2":
    objeto = input("Ingrese objeto (car, bicycle, traffic light, person): ")
    objetos_seleccionados = [objeto]
elif opcion == "3":
    objetos = input("Ingrese objetos separados por coma (car,bicycle,person): ")
    objetos_seleccionados = [o.strip() for o in objetos.split(",")]
else:
    exit()

carpeta_videos = "carpeta_videos"
videos = [f for f in os.listdir(carpeta_videos) if f.endswith((".mp4", ".avi"))]

for video in videos:
    ruta = os.path.join(carpeta_videos, video)
    cap = cv2.VideoCapture(ruta)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resultados = model(frame, verbose=False)

        for r in resultados:
            for box in r.boxes:
                clase_id = int(box.cls[0])
                nombre = model.names[clase_id]

                if nombre in objetos_seleccionados:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = colores.get(nombre, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, nombre, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Reconocimiento", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

cv2.destroyAllWindows()