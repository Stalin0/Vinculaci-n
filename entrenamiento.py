import cv2
import os
import numpy as np
import time

def obtenerModelo(method, facesData, labels):
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFFaceRecognizer_create()


    print("El modelo de entrenamiento( "+method+")...")
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time()-inicio
    print("El timepo de entrenamiento es ( "+method+" ): ", tiempoEntrenamiento)


    emotion_recognizer.write("modelo"+method+".xml")

dataPath = 'C:/Users/User/Documents/ESPE/Vinculación/Data'
emotionsList = os.listdir(dataPath)
print('Lista de las personas: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
    emotionsPath = dataPath + '/' + nameDir

    for fileName in os.listdir(emotionsPath):
        labels.append(label)
        facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))

    label =  label + 1

obtenerModelo('EigenFaces', facesData, labels)
obtenerModelo('FisherFaces', facesData, labels)
obtenerModelo('LBPH', facesData, labels)
        