
from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np, mediapipe as mp, io, os
from PIL import Image
from deepface import DeepFace
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def clasificar_rostro(face):
    p_izq, p_der = face.landmark[234], face.landmark[454]
    jaw_izq, jaw_der = face.landmark[172], face.landmark[397]
    frente, barbilla = face.landmark[10], face.landmark[152]

    ancho_pomulos = abs(p_der.x - p_izq.x)
    ancho_mandibula = abs(jaw_der.x - jaw_izq.x)
    alto_rostro = abs(barbilla.y - frente.y)
    proporcion = alto_rostro / ancho_pomulos

    menton_tipo = "puntiagudo" if barbilla.y > 0.75 else "redondeado"

    if proporcion > 1.6 and ancho_mandibula < ancho_pomulos:
        forma = "ovalado"
    elif proporcion < 1.3 and abs(ancho_mandibula - ancho_pomulos) < 0.02:
        forma = "redondo"
    elif ancho_mandibula > ancho_pomulos * 1.1:
        forma = "cuadrado"
    elif proporcion > 1.6 and ancho_mandibula > ancho_pomulos:
        forma = "rectangular"
    elif ancho_pomulos > ancho_mandibula and menton_tipo == "puntiagudo":
        forma = "corazon"
    elif ancho_pomulos > ancho_mandibula:
        forma = "diamante"
    else:
        forma = "ovalado"

    return forma, menton_tipo

def largo_cuello(face):
    menton = face.landmark[152]
    cuello = face.landmark[199]
    dist = abs(cuello.y - menton.y)
    if dist > 0.15: return "largo"
    elif dist < 0.10: return "corto"
    return "medio"

def tono_piel(image, face):
    h,w,_ = image.shape
    mejilla = face.landmark[234]
    x,y = int(mejilla.x*w), int(mejilla.y*h)
    zona = image[y-20:y+20, x-20:x+20]
    b,g,r = np.mean(zona,axis=(0,1))
    return "calida" if r>b else "fria"

def detectar_edad(path):
    try:
        edad = DeepFace.analyze(path, actions=['age'], enforce_detection=False)[0]['age']
    except:
        edad = 30
    return "joven" if edad<30 else "madura"

def tipo_cabello(face):
    oreja_izq = face.landmark[234]
    oreja_der = face.landmark[454]
    frente = face.landmark[10]
    if oreja_izq.y > frente.y and oreja_der.y > frente.y:
        return "suelto"
    elif oreja_izq.y < frente.y and oreja_der.y < frente.y:
        return "recogido"
    return "corto"

@app.post("/analizar")
async def analizar(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img.save("temp.jpg")
    image = np.array(img)

    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    face = results.multi_face_landmarks[0]

    rostro, menton = clasificar_rostro(face)
    cuello = largo_cuello(face)
    tono = tono_piel(image, face)
    edad = detectar_edad("temp.jpg")
    cabello = tipo_cabello(face)

    prompt=f"""
    Clienta:
    Rostro {rostro}, mentón {menton}, cuello {cuello},
    piel {tono}, edad {edad}, cabello {cabello}.
    Recomienda aretes ideales.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return {"rostro":rostro,"menton":menton,"cuello":cuello,
            "tono":tono,"edad":edad,"cabello":cabello,
            "recomendacion":resp.choices[0].message.content}
