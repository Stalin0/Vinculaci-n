import cv2
import os
import imutils
import tkinter as tk
from tkinter import messagebox

#emotion = 'Enojo'
#emotion = 'Felicidad'
#emotion = 'Sorpresa'
#emotion = 'Tristeza'
#emotion = 'Neutral'
emotion = 'Disgusto'

dataPath = 'C:/Users/User/Documents/GitHub/Vinculacion/Data'
emotionsPath = dataPath + '/' + emotion

if not os.path.exists(emotionsPath):
    print('Se creo la carpeta correctamente...', emotionsPath)
    os.makedirs(emotionsPath)
#Hacemos la captura con la camara para la detección del rostro
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:

    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(emotionsPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >=200:
        break
    
cap.release()
cv2.destroyAllWindows()
root = tk.Tk()
root.withdraw()
messagebox.showinfo("Éxito", "Proceso completado con éxito")
root.destroy()