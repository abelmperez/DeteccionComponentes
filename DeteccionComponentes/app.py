from flask import Flask, request, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


def align_images(img_ref, img_test):
    """Alinear img_test con img_ref usando SIFT o rotaciones discretas."""
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    # Intentar alineación con SIFT
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_test, descriptors_test = sift.detectAndCompute(gray_test, None)

    if descriptors_ref is None or descriptors_test is None:
        print("No se encontraron suficientes puntos clave para SIFT")
        # Como respaldo, usar redimensionamiento simple
        return cv2.resize(img_test, (img_ref.shape[1], img_ref.shape[0]))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_ref, descriptors_test, k=2)

    # Filtrar coincidencias con el test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img_ref.shape[:2]
        img_test_aligned = cv2.warpPerspective(img_test, M, (w, h))
        return img_test_aligned

    # Si SIFT falla, probar rotaciones discretas (0°, 90°, 180°, 270°)
    print("SIFT no encontró suficientes coincidencias, probando rotaciones discretas")
    best_img = img_test
    best_diff = float('inf')
    h, w = img_ref.shape[:2]

    for angle in [0, 90, 180, 270]:
        # Rotar la imagen
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_rotated = cv2.warpAffine(img_test, M, (w, h))

        # Calcular diferencia con la imagen de referencia
        gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_ref, gray_rotated)
        max_diff = np.max(diff)

        if max_diff < best_diff:
            best_diff = max_diff
            best_img = img_rotated

    return best_img


@app.route('/upload', methods=['POST'])
def upload():
    if 'referencia' not in request.files or 'prueba' not in request.files:
        return "Error: No se seleccionaron ambas imágenes", 400

    referencia = request.files['referencia']
    prueba = request.files['prueba']

    if referencia.filename == '' or prueba.filename == '':
        return "Error: Uno o ambos archivos no tienen nombre", 400

    ref_path = os.path.join(UPLOAD_FOLDER, 'referencia.jpg')
    prueba_path = os.path.join(UPLOAD_FOLDER, 'prueba.jpg')
    referencia.save(ref_path)
    prueba.save(prueba_path)

    if not os.path.exists(ref_path) or not os.path.exists(prueba_path):
        return f"Error: No se pudieron guardar las imágenes en {UPLOAD_FOLDER}", 400

    # Cargar las imágenes
    img_referencia = cv2.imread(ref_path)
    img_prueba = cv2.imread(prueba_path)

    if img_referencia is None or img_prueba is None:
        return f"Error: No se pudo cargar una o ambas imágenes. Verifica las rutas: {ref_path}, {prueba_path}", 400

    # Alinear la imagen de prueba
    img_prueba_alineada = align_images(img_referencia, img_prueba)

    # Guardar la imagen alineada para inspección
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'prueba_alineada.jpg'), img_prueba_alineada)

    # Convertir a escala de grises para comparación
    gray_referencia = cv2.cvtColor(img_referencia, cv2.COLOR_BGR2GRAY)
    gray_prueba_alineada = cv2.cvtColor(img_prueba_alineada, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque
    gray_referencia = cv2.GaussianBlur(gray_referencia, (7, 7), 0)
    gray_prueba_alineada = cv2.GaussianBlur(gray_prueba_alineada, (7, 7), 0)

    # Calcular la diferencia absoluta
    diferencia = cv2.absdiff(gray_referencia, gray_prueba_alineada)

    # Guardar la imagen de diferencia
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'diferencia.jpg'), diferencia)

    # Verificar si las imágenes son idénticas
    max_diferencia = np.max(diferencia)
    print(f"Valor máximo de diferencia: {max_diferencia}")

    if max_diferencia < 10:
        return render_template('result.html', resultado='uploads/resultado.jpg',
                               mensaje="Las imágenes son idénticas, no se detectaron componentes faltantes")

    # Umbral para detectar diferencias
    _, thresh = cv2.threshold(diferencia, 100, 255, cv2.THRESH_BINARY)

    # Dilatar
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_resultado = img_prueba_alineada.copy()
    componentes_faltantes = False
    for contorno in contornos:
        if cv2.contourArea(contorno) > 100:
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 0, 255), 2)
            componentes_faltantes = True

    resultado_path = os.path.join(UPLOAD_FOLDER, 'resultado.jpg')
    cv2.imwrite(resultado_path, img_resultado)

    if not componentes_faltantes:
        return render_template('result.html', resultado='uploads/resultado.jpg',
                               mensaje="No se detectaron componentes faltantes")

    return render_template('result.html', resultado='uploads/resultado.jpg', mensaje="Componentes faltantes detectados")


if __name__ == '__main__':
    app.run(debug=True)