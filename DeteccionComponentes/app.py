from flask import Flask, render_template
import cv2
import numpy as np
import os
import time
from datetime import datetime
import threading
import glob

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static/uploads'
INPUT_FOLDER = 'input_images'  # Carpeta para imágenes de la cámara
REFERENCE_IMAGE = os.path.join(UPLOAD_FOLDER, 'referencia.jpg')
SAVE_IMAGES = True  # Configurable: True para guardar imágenes, False para no guardar
CHECK_INTERVAL = 5  # Intervalo en segundos para verificar nuevas imágenes

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INPUT_FOLDER, exist_ok=True)

def align_images(img_ref, img_test):
    """Alinear img_test con img_ref usando SIFT o rotaciones discretas."""
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    # Intentar alineación con SIFT
    sift = cv2.SIFT_create(nfeatures=1000)  # Más puntos clave para rotaciones grandes
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_test, descriptors_test = sift.detectAndCompute(gray_test, None)

    if descriptors_ref is None or descriptors_test is None:
        print("No se encontraron suficientes puntos clave para SIFT")
        return cv2.resize(img_test, (img_ref.shape[1], img_ref.shape[0]))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_ref, descriptors_test, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 5:  # Reducido para mejorar detección
        src_pts = np.float32([keypoints_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img_ref.shape[:2]
        img_test_aligned = cv2.warpPerspective(img_test, M, (w, h))
        return img_test_aligned

    print("SIFT no encontró suficientes coincidencias, probando rotaciones discretas")
    best_img = img_test
    best_diff = float('inf')
    h, w = img_ref.shape[:2]

    for angle in [0, 90, 180, 270]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_rotated = cv2.warpAffine(img_test, M, (w, h))
        gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_ref, gray_rotated)
        max_diff = np.max(diff)

        if max_diff < best_diff:
            best_diff = max_diff
            best_img = img_rotated

    return best_img

def process_image(img_path, img_ref):
    """Procesar una imagen capturada y compararla con la referencia."""
    img_test = cv2.imread(img_path)
    if img_test is None:
        print(f"Error: No se pudo cargar la imagen {img_path}")
        return None, None, None

    # Alinear la imagen
    img_test_aligned = align_images(img_ref, img_test)

    # Guardar imagen alineada (opcional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aligned_path = os.path.join(UPLOAD_FOLDER, f'prueba_alineada_{timestamp}.jpg')
    if SAVE_IMAGES:
        cv2.imwrite(aligned_path, img_test_aligned)

    # Convertir a escala de grises
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_test_aligned = cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque
    gray_ref = cv2.GaussianBlur(gray_ref, (7, 7), 0)
    gray_test_aligned = cv2.GaussianBlur(gray_test_aligned, (7, 7), 0)

    # Calcular diferencia
    diferencia = cv2.absdiff(gray_ref, gray_test_aligned)

    # Guardar diferencia (opcional)
    diff_path = os.path.join(UPLOAD_FOLDER, f'diferencia_{timestamp}.jpg')
    if SAVE_IMAGES:
        cv2.imwrite(diff_path, diferencia)

    # Verificar si las imágenes son idénticas
    max_diferencia = np.max(diferencia)
    print(f"Valor máximo de diferencia para {img_path}: {max_diferencia}")

    if max_diferencia < 10:
        return img_test_aligned, "Las imágenes son idénticas, no se detectaron fallos", None

    # Umbral para detectar diferencias
    _, thresh = cv2.threshold(diferencia, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_resultado = img_test_aligned.copy()
    fallos_detectados = False
    for contorno in contornos:
        if cv2.contourArea(contorno) > 100:
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fallos_detectados = True

    # Guardar resultado (opcional)
    result_path = os.path.join(UPLOAD_FOLDER, f'resultado_{timestamp}.jpg')
    if SAVE_IMAGES:
        cv2.imwrite(result_path, img_resultado)

    mensaje = "Fallos detectados" if fallos_detectados else "No se detectaron fallos"
    return img_resultado, mensaje, result_path

def check_new_images():
    """Verificar nuevas imágenes en input_images y procesarlas."""
    img_ref = cv2.imread(REFERENCE_IMAGE)
    if img_ref is None:
        print(f"Error: No se pudo cargar la imagen de referencia {REFERENCE_IMAGE}")
        return

    #cap = cv2.VideoCapture(0)  # Cámara por defecto
    #while True:
    #    ret, frame = cap.read()
    #    if ret:
    #        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #        img_path = os.path.join(INPUT_FOLDER, f'captura_{timestamp}.jpg')
    #        cv2.imwrite(img_path, frame)
    #    time.sleep(CHECK_INTERVAL)
    #cap.release()**/

    processed_images = set()  # Para evitar procesar la misma imagen varias veces
    while True:
        for img_path in glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')):
            if img_path not in processed_images:
                print(f"Procesando nueva imagen: {img_path}")
                processed_images.add(img_path)
                img_resultado, mensaje, result_path = process_image(img_path, img_ref)
                if img_resultado is not None:
                    print(f"Resultado para {img_path}: {mensaje}")
                    # Opcional: mover la imagen procesada para evitar reprocesamiento
                    if SAVE_IMAGES:
                        os.rename(img_path, os.path.join(UPLOAD_FOLDER, f"capturada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"))
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    # Mostrar los últimos resultados (basado en archivos en static/uploads)
    result_images = [f'uploads/resultado_{f}' for f in os.listdir(UPLOAD_FOLDER) if f.startswith('resultado_')]
    result_images.sort(reverse=True)  # Últimos resultados primero
    mensajes = []
    for img in result_images[:5]:  # Mostrar hasta 5 resultados
        timestamp = img.split('resultado_')[1].replace('.jpg', '')
        mensajes.append(f"Resultado {timestamp}: {'Fallos detectados' if os.path.exists(os.path.join(UPLOAD_FOLDER, f'diferencia_{timestamp}.jpg')) else 'No se detectaron fallos'}")
    return render_template('results.html', results=zip(result_images, mensajes))

if __name__ == '__main__':
    # Iniciar el procesamiento automático en un hilo separado
    threading.Thread(target=check_new_images, daemon=True).start()
    app.run(debug=True)