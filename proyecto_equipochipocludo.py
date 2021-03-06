import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
from os import scandir
import os
from sklearn import svm
from tkinter import *
from tkinter import ttk, messagebox, filedialog
import tkinter as tk
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import cv2

filename = None

raiz = Tk()
raiz.title("Proyecto Final Detección de vocales Eq. Chipocludo")
raiz.resizable(0,0)
raiz.geometry("500x350")
raiz.config(bg="cyan")

myFrame=Frame()
myFrame.pack(side="top")
myFrame.config(bg="white")

#Logo ESCOM
imagen=tk.PhotoImage(file="Imagenes GUI/logoescom.png")
imagen_sub=imagen.subsample(12)
widget=ttk.Label(image=imagen_sub)
widget.place(x=5,y=5)

#Logo IPN
imageni=tk.PhotoImage(file="Imagenes GUI/ipn.png")
imageni_sub=imageni.subsample(15)
widgeti=ttk.Label(image=imageni_sub)
widgeti.place(x=400,y=5)


text = Label(text="Escuela Superior de Cómputo\n\n Arévalo Andrade Miguel Ángel \n Esquivel Salvatti José Luis\n \
            López Morales Miguel Ángel\n Vaca García Jesús Fernando\n Vargas Espino Carlos Hassan")
text.place(x=155,y=7)

def DFT_slow(x):
    """Calcula la transformada discreta de Fourier del arreglo x"""
    x = np.asarray(x, dtype=float) # Vector de valores de frecuencia
    N = x.shape[0]  #Regresa la dimension de la matriz de frecuencias
    n = np.arange(N) #Devuelve valores espaciados uniformemente dentro de un intervalo dado.
    k = n.reshape((N, 1)) #Redimensionar de N a 1
    M = np.exp(-2j * np.pi * k * n / N)  #Definicion de la DFT
    return np.dot(M, x) #Devolver el producto punto de los dos arreglos

def FFT(x):
    """Una implementación recursiva de la FFT 1D Cooley-Tukey"""
    x = np.asarray(x, dtype=float) #Componentes del vector x convertir a un vector de tipo flotante
    N = x.shape[0] #Dimensión del vector dada la restriccion del algoritmo
    
    if N % 2 > 0: #Restringir a potencias mayores de 2 (Caso Base)
        raise ValueError("el tamaño de x debe ser una potencia de 2")
    elif N <= 32: 
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2]) #Llamada recursiva con los valores pares
        X_odd = FFT(x[1::2]) #Llamada recursiva con los valores impares
        factor = np.exp(-2j * np.pi * np.arange(N) / N) # Valores exponenciales de la suma
        return np.concatenate([X_even + factor[:N / 2] * X_odd, #Concatenacion resultante de las referencias par e impar de x
                               X_even + factor[N / 2:] * X_odd])

def make_x_y(labels):
    '''Lee los archivos wav en una carpeta aplica fft y guarda la componenete de mayor frecuencia, 
    regresa el arreglo con dicha componente y la etiqueta de cada cada archivo'''
    x,y = [],[] #Listas auxiliares para guardar las componentes
    i=0
    for label in labels: #Para cada ruta se leen todos los archivos de la carpeta
        files = lsi(str(os.getcwd())+'/'+label)
        for file in files: #Para cada archivo 
            rate, data = wav.read(str(os.getcwd())+'/'+label+'/'+file)
            data = np.setdiff1d(data,0) #Devuelve la informacion valiosa del audio quitando los ceros
            data = np.array(data[:32]) #Crear un arreglo de 32
            fft_out = FFT(data) #Hacer FFT
            fft_mag=np.absolute(fft_out) #Valor absoluto del arreglo anterior
            mf=np.where(fft_mag==np.amax(fft_mag)) #Guardar valores de frecuencia máximos
            comp=fft_out[mf] #Posición de la frecuencia maxima se guarda en comp
            x.append(comp)  #Se agregan a las listas auxiliares
            y.append(label)
        i+=1
    return x,y

        
def make_model(x,y):
    '''Crea un modelo de ML basado en el algoritmo de clasificacion de la Máquia de Soporte Vectorial
    regresa el modelo entrenado'''
    clf = svm.SVC(gamma='auto')
    clf.fit(x, y)  
    return clf 

def make_prediction_svm(model,fname):
    global filename
    '''Hace una prediccion recibiendo el modelo entrenado y el nombre del archivo a leer
    por modelos de sckit learn se tiene que hacer dos predicciones pero solo retorna el valor a predecir'''
    rate, data = wav.read(fname) #Leer los archivos y se separa en dos variables
    data = np.setdiff1d(data,0) #Se eliminan los ceros
    data = np.array(data[:32]) #Se redimensiona a tamaño 32
    fft_out = FFT(data)  #Ejecutar FFT
    fft_mag=np.absolute(fft_out)  #Valor absoluto del arreglo anterior
    mf=np.where(fft_mag==np.amax(fft_mag)) #Guardar valores de frecuencia máximos
    comp=fft_out[mf]#Posición de la frecuencia maxima se guarda en comp
    r= float(np.real(comp)) #Se separan las componentes reales
    i = float(np.imag(comp)) #Se separan las componentes imaginarias
    vec1,vec=[],[] #Una lista de vectores 
    vec1.append(r) # Se agregan a la lista
    vec1.append(i) # Se agregan a la lista 
    vec.append(vec1) #Se agrega el vector a la lista
    rate, data = wav.read(filename)
    data = np.setdiff1d(data,0)
    data = np.array(data[:32])
    fft_out = FFT(data)
    fft_mag=np.absolute(fft_out)
    mf=np.where(fft_mag==np.amax(fft_mag)) 
    comp=fft_out[mf]
    vec1=[]
    r= float(np.real(comp))
    i = float(np.imag(comp))
    vec1.append(r)
    vec1.append(i)
    vec.append(vec1)
    vec = np.asarray(vec)
    aux = model.predict(vec)[0]
    return aux[-1]

    
def make_X(x,y):
    '''crea la matriz X recibiendo las componenetes reales e imaginarias por separado'''
    New_X=[]
    for i in range(len(y)):
        new_X=np.append(x[i], y[i])
        New_X.append(new_X)
    New_X = np.asarray(New_X)
    return New_X
            
def lsi(path):
    '''Lee todos los archivos dentro de path'''
    return([obj.name for obj in scandir(path) if obj.is_file()])

def abrirArchivo_a_Usar():
    global filename
    filename = filedialog.askopenfilename(initialdir="C:",title = "Selecciona un archivo.wav para predecir",filetypes=(("wav files","*.wav"),("all files","*.*")))
    print(filename)
    # head, tail = os.path.split(filename)
    # filename = tail
    # print(filename)

def grabarAudio():
    # Sampling frequency
    freq = 48100
    
    # Recording duration
    duration = 2
    
    # Start recorder with the given values 
    # of duration and sample frequency
    print("Ya estoy grabando")
    recording = sd.rec(int(duration * freq), 
                    samplerate=freq, channels=2)
    
    # Record audio for the given number of seconds
    sd.wait()
   
    
    # Convert the NumPy array to audio file
    wv.write("Grabacion_actual.wav", recording, freq, sampwidth=2)

def ejecutarProceso():
    x,y=make_x_y(['Audios/Vocales/A','Audios/Vocales/E','Audios/Vocales/I','Audios/Vocales/O','Audios/Vocales/U'])
    r= np.real(x)#obtiene las componenetes reales
    i = np.imag(x)#obtiene las componenetes imaginarias
    X = make_X(r,i)
    modelV=make_model(X,y)#entrena el modelo para predecir vocales
    messagebox.showinfo(title= "Reconocimiento", message = f'Letra {make_prediction_svm(modelV,filename).lower()} reconocida' )
    xs,ys=make_x_y(['Audios/Genero/H','Audios/Genero/M'])
    rs= np.real(xs)#obtiene las componenetes reales
    iS = np.imag(xs)#obtiene las componenetes imaginaria
    XS = make_X(rs,iS)
    modelSexo=make_model(XS,ys)#entrena el modelo para predecir sexos
    print(make_prediction_svm(modelSexo,filename))
    image = ""
    if make_prediction_svm(modelSexo,filename) == 'H':
        image = 'Imagenes GUI/hombre.png'
    elif make_prediction_svm(modelSexo,filename) == 'M':
        image = 'Imagenes GUI/mujer.png'
    
    img = cv2.imread(image)
    img_resized = cv2.resize(img, (350, 280)) 
    cv2.imshow('Genero detectado',img_resized)

    

abrir=Button(raiz, text="Seleccionar archivo de audio",command=abrirArchivo_a_Usar)
abrir.place(x=25,y=130)

start=Button(raiz, text="Ejecutar",command = ejecutarProceso)
start.place(x=150,y=300)  

record_audio=Button(raiz, text="Grabar audio",command = grabarAudio)
record_audio.place(x=250,y=300) 

raiz.mainloop()
