import os
import tempfile
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image  

# =========================
# PATCH TORCH (IMPORTANTE)
# =========================
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# =========================
# APP FLASK
# =========================
app = Flask(__name__)

MODELO = "Epp_RN.pt"

print("Cargando modelo...")
model = YOLO(MODELO, task="detect")
print("Modelo cargado correctamente")


# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.route("/test", methods=["POST"])
def test():

    foto = request.files.get("imagen")

    if foto is None:
        return jsonify({
            "estado": "error",
            "mensaje": "No se envió imagen"
        }), 400

    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        ruta = temp.name

    foto.save(ruta)

    # =========================
    # 🔥 FIX IMPORTANTE (CAMARA / SIMULADOR)
    # =========================
    try:
        img = Image.open(ruta)
        img = img.convert("RGB")
        img.save(ruta, quality=100)
    except Exception as e:
        print("Error procesando imagen:", e)

    try:

        # =========================
        # PREDICCIÓN YOLO
        # =========================
        results = model.predict(
            source=ruta,
            conf=0.25,
            save=False,
            verbose=False
        )

        detecciones = []

        personas = 0
        epp_detectado = 0

        print("\n========== DETECCIONES ==========")

        for box in results[0].boxes:

            clase_id = int(box.cls.item())
            nombre = model.names[clase_id]
            confianza = round(float(box.conf.item()), 3)

            print(f"{nombre} ({confianza})")

            detecciones.append({
                "clase": nombre,
                "confianza": confianza
            })

            # =========================
            # CONTADORES LÓGICOS
            # =========================
            if nombre.lower() == "person":
                personas += 1

            if nombre.lower() in ["helmet", "glove", "vest", "epp"]:
                epp_detectado += 1

        print("=================================\n")

        # borrar archivo temporal
        if os.path.exists(ruta):
            os.remove(ruta)

        # =========================
        # LÓGICA DE DECISIÓN FINAL
        # =========================

        if personas == 0:

            estado = "sin_persona"
            mensaje = "No se identificó persona en la imagen"

        elif personas > 0 and epp_detectado == 0:

            estado = "infraccion"
            mensaje = "Infracción EPP detectada"

        else:

            estado = "ok"
            mensaje = "EPP detectado correctamente"

        # =========================
        # 📦 RESPUESTA FINAL
        # =========================
        return jsonify({
            "estado": estado,
            "mensaje": mensaje,
            "personas": personas,
            "total_detecciones": len(detecciones),
            "detecciones": detecciones
        })

    except Exception as e:

        if os.path.exists(ruta):
            os.remove(ruta)

        return jsonify({
            "estado": "error",
            "mensaje": str(e)
        }), 500


# =========================
# CLASES DEL MODELO
# =========================
@app.route("/clases", methods=["GET"])
def clases():
    return jsonify(model.names)


# =========================
#  RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )