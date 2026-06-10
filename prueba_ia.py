import torch
import ultralytics
from ultralytics import YOLO
import os

# 1. Evitar advertencias de seguridad estrictas de PyTorch al cargar pesos locales
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

archivo_modelo = "Epp_RN.pt"

# 2. Verificar si el archivo se descargó correctamente
if not os.path.exists(archivo_modelo):
    print(f"Error: No encuentro el archivo '{archivo_modelo}' en esta carpeta.")
    print("Por favor, ejecSuta primero el comando de descarga (Invoke-WebRequest) en tu terminal.")
else:
    print("Cargando el modelo local nativo de IA especializado en EPP...")
    # Cargamos directamente el archivo local sin intermediarios inestables
    model = YOLO(archivo_modelo, task="detect")

    print("\n📋 CLASES Y RESTRICCIONES DETECTADAS:")
    for id_clase, nombre in model.names.items():
        print(f"   🔹 ID {id_clase}: {nombre}")

    # Imagen de prueba de internet para verificar que el motor procesa
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

    print("\n🧠 Ejecutando predicción de la Red Neuronal...")
    results = model.predict(source=image, conf=0.25, save=False)

    print("\n📦 Información de cajas de detección encontradas:")
    print(results[0].boxes)

    print("\n🖼️ ¡Proceso completado con éxito! La IA está operando de forma nativa.")