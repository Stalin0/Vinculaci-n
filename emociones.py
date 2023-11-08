import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageFont, ImageDraw, Image
import time

def emotionImage(emotion):
    # Función para cargar imágenes de emojis basadas en la emoción
    if emotion == 'Felicidad':
        image = cv2.imread('Emojis/felicidad.jpg')
    if emotion == 'Enojo':
        image = cv2.imread('Emojis/enojo.jpg')
    if emotion == 'Sorpresa':
        image = cv2.imread('Emojis/sorpresa.jpg')
    if emotion == 'Tristeza':
        image = cv2.imread('Emojis/tristeza.jpg')
    if emotion == 'Disgusto':
        image = cv2.imread('Emojis/disgusto.jpg')
    if emotion == 'Neutral':
        image = cv2.imread('Emojis/neutral.jpg')
    return image

method = 'EigenFaces'
#method = 'FisherFaces'
#method = 'LBPH'  
# Crear el reconocedor basado en el método seleccionado
if method == 'EigenFaces':
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
elif method == 'FisherFaces':
    emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
elif method == 'LBPH':
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    # Mostrar una alerta de error si se selecciona un método no válido
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Error", "Método de reconocimiento no válido")
    root.destroy()
    exit()


modelo_path = 'modelo' + method + '.xml'
if os.path.exists(modelo_path):
    emotion_recognizer.read(modelo_path)
else:
    
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Error", "No se pudo cargar el modelo " + method)
    root.destroy()
    exit()

dataPath = 'C:/Users/User/Documents/ESPE/Vinculacion/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])
 


    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        if method == 'EigenFaces':
            if result[1] < 5700:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])
        if method == 'FisherFaces':
            if result[1] < 500:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])
        if method == 'LBPH':
            if result[1] < 60:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    cv2.imshow('nFrame', nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
