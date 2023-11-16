import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class EmotionCapture:
    def __init__(self, master):
        self.master = master
        self.master.title("Seleccionar Emoción")
        self.master.geometry("800x600")  

        self.main_frame = tk.Frame(self.master, width=800, height=600)  
        self.main_frame.pack(fill="both", expand=True)

        self.emotions = ['Enojo', 'Felicidad', 'Sorpresa', 'Tristeza', 'Neutral', 'Disgusto']
        self.selected_emotion = tk.StringVar(self.master, value=self.emotions[0])

        self.label = tk.Label(self.main_frame, text="Selecciona la emoción:")
        self.label.pack()

        self.emotion_menu = ttk.Combobox(self.main_frame, textvariable=self.selected_emotion, values=self.emotions, width=20)
        self.emotion_menu.pack()

        self.start_button = tk.Button(self.main_frame, text="Iniciar Captura", command=self.start_capture)
        self.start_button.pack()

        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack()

        self.data_path = 'C:/Users/User/Documents/GitHub/Vinculacion/Data'

    def start_capture(self):
        emotion = self.selected_emotion.get()
        emotions_path = os.path.join(self.data_path, emotion)

        if not os.path.exists(emotions_path):
            print('Se creó la carpeta correctamente...', emotions_path)
            os.makedirs(emotions_path)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.count = 0

        self.capture(emotions_path)

    def capture(self, emotions_path):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))  # Ajusta el tamaño según tus necesidades
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aux_frame = frame.copy()

            faces = self.face_classif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = aux_frame[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(emotions_path + '/rostro_{}.jpg'.format(self.count), face_roi)
                self.count += 1

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)

            self.video_label.img = img
            self.video_label.config(image=img)

            if self.count < 200:
                self.master.after(10, self.capture, emotions_path)  # Llamada recursiva después de 10 milisegundos
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                self.show_success_message()

    def show_success_message(self):
        messagebox.showinfo("Éxito", "Proceso completado con éxito")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionCapture(root)
    root.mainloop()
