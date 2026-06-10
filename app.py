
import os
import torch
import tempfile
from flask import Flask, request, jsonify, send_from_directory
import psycopg2
from ultralytics import YOLO
from datetime import datetime
import requests

# --- CONFIGURACIÓN ---
IMG_BB_KEY = "d3e8627d7e33547f6135a52ed114fb34"
# Parche de seguridad para PyTorch
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

app = Flask(__name__)
# Diccionarios
TRADUCCIONES = {
    'person': 'persona',
    'helmet': 'casco',
    'glove': 'guante',
    'goggles': 'gafa',
    'mask': 'mascarilla',
    'vest': 'chaleco'
}

DB_URL = "postgresql://bd_seguridad_epp_user:8OdHgv8EKafcMPTnNUfWvLnu55SwBFRk@dpg-d8bsno58nd3s738v15i0-a.oregon-postgres.render.com/bd_seguridad_epp"
ARCHIVO_MODELO = "Epp_RN.pt"

# Cargar IA
if os.path.exists(ARCHIVO_MODELO):
    print("Cargando Red Neuronal EPP...")
    model = YOLO(ARCHIVO_MODELO, task="detect")
else:
    model = None
    print("ERROR: Modelo no encontrado.")

def obtener_conexion():
    return psycopg2.connect(DB_URL)

# Función para subir a ImgBB
def subir_a_imgbb(ruta_imagen):
    try:
        # Abrimos el archivo en modo lectura binaria
        with open(ruta_imagen, "rb") as file:
            # Enviamos el archivo tal cual como "image"
            payload = {"key": IMG_BB_KEY}
            files = {"image": file}
            res = requests.post("https://api.imgbb.com/1/upload", data=payload, files=files)
            
            # Verificamos si hubo éxito
            if res.status_code == 200:
                return res.json()['data']['url']
            else:
                print(f"Error en ImgBB ({res.status_code}): {res.text}")
                return "https://via.placeholder.com/150?text=Error+Subida"
    except Exception as e:
        print(f"ERROR CRÍTICO subiendo a ImgBB: {e}")
        return "https://via.placeholder.com/150?text=Error+Critico"
    
@app.route('/obtener_camaras', methods=['GET'])
def obtener_camaras():
    try:
        conn = obtener_conexion()
        cur = conn.cursor()
        cur.execute("""
            SELECT c.codigo_camara, s.nombre_salon
            FROM camaras c
            JOIN salones s ON c.id_salon = s.id_salon
        """)
        filas = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify([f"{fila[0]} ({fila[1]})" for fila in filas]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/simular_camara', methods=['POST'])
def simular_camara():

    if model is None:
        return jsonify({"error": "El motor de IA no está cargado"}), 500

    codigo_camara = request.form.get("codigo_camara")
    foto_archivo = request.files.get('imagen')

    if not codigo_camara or not foto_archivo:
        return jsonify({"error": "Faltan datos"}), 400

    # guardar temporal
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        ruta_temporal = temp_file.name

    foto_archivo.save(ruta_temporal)

    conn = None
    cur = None

    try:
        conn = obtener_conexion()
        cur = conn.cursor()

        dias = {
            0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
            4: "Viernes", 5: "Sábado", 6: "Domingo"
        }

        dia_actual = dias[datetime.now().weekday()].strip().capitalize()
        hora_actual = datetime.now().time()

        cur.execute("""
            SELECT h.id_docente
            FROM horarios_clases h
            JOIN camaras c ON h.id_salon = c.id_salon
            WHERE c.codigo_camara = %s
            AND h.dia_semana = %s
            AND %s BETWEEN h.hora_inicio AND h.hora_fin
        """, (codigo_camara, dia_actual, hora_actual))

        horario_reg = cur.fetchone()
        id_docente = horario_reg[0] if horario_reg else None

        # --- IA ---
        results = model.predict(source=ruta_temporal, conf=0.25, save=False, verbose=False)

        conteos = {'person': 0, 'helmet': 0, 'glove': 0, 'goggles': 0, 'mask': 0, 'vest': 0}
        objetos_detectados = []

        for box in results[0].boxes:
            nombre = model.names[int(box.cls.item())].lower()
            if nombre in conteos:
                conteos[nombre] += 1
                objetos_detectados.append(nombre)

        personas = conteos['person']

        infraccion_detectada = False
        descripcion = None
        severidad = None

        # No hay personas
        if personas == 0:

            print("No se detectaron personas")

            return jsonify({
                "estado": "Procesado",
                "infraccion": False,
                "motivo": "No se detectaron personas",
                "docente_notificado": False
            }), 200

        faltas = []

        for clase, cantidad in conteos.items():

            if clase == "person":
                continue

            if cantidad < personas:

                nombre_es = TRADUCCIONES.get(clase, clase)

                faltas.append(
                    f"{personas - cantidad} {nombre_es}(s)"
                )

        # EPP completo
        if len(faltas) == 0:

            return jsonify({
                "estado": "Procesado",
                "infraccion": False,
                "motivo": "EPP completo",
                "docente_notificado": False
            }), 200

        # Hay incumplimiento
        infraccion_detectada = True

        total_epp = (
            conteos['helmet']
            + conteos['glove']
            + conteos['goggles']
            + conteos['mask']
            + conteos['vest']
        )

        if total_epp == 0:

            severidad = "ALTA"

            descripcion = (
                "Se detectaron incumplimientos críticos "
                "de seguridad en el salón. "
                "Por favor revise la situación."
            )

        else:

            severidad = "MEDIA"

            descripcion = (
                "Se detectaron incumplimientos de seguridad "
                "en el salón. "
                "Por favor revise la situación."
            )

        print("\n========== ALERTA ==========")
        print("Camara:", codigo_camara)
        print("Severidad:", severidad)
        print("Detalle:", ", ".join(faltas))
        print("============================\n")

        if infraccion_detectada:
            try:
                requests.post("http://192.168.127.210:3000/api/notificacion-flask", json={
                    "codigo_camara": codigo_camara,
                    "tipo_falta": descripcion,
                    "severidad": severidad
                })
            except Exception as e:
                print("Error notificando Node:", e)

        # --- GUARDADO ---
        if infraccion_detectada and id_docente not in (None, "", 0, "0"):
            id_docente = int(id_docente)

            # Subir a ImgBB
            url_evidencia = subir_a_imgbb(ruta_temporal)

            cur.execute("""
                INSERT INTO alertas (codigo_camara, id_docente, tipo_falta, severidad, fecha_hora, evidencia_url)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (codigo_camara, id_docente, descripcion, severidad,datetime.now(), url_evidencia))

            conn.commit()
            print("✔ Alerta guardada en BD con URL pública")

        return jsonify({
            "estado": "Procesado",
            "infraccion": infraccion_detectada,
            "motivo": descripcion,
            "docente_notificado": bool(id_docente)
        }), 200

    except Exception as e:
        print(f"Error crítico: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
            if os.path.exists(ruta_temporal): os.remove(ruta_temporal)
        except: pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
