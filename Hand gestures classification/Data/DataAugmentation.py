import os
import numpy as np
import csv
from random import randrange

data_dir = 'C:/Users/penad/Desktop/MUIA/TFM/Logger'
tag_dir = 'C:/Users/penad/Desktop/MUIA/TFM/Logger/Data'
fname = os.path.join(tag_dir, 'LabelsDataAug.csv')
fnameData = os.path.join(data_dir, 'DataArduino.csv')
fnameTags = os.path.join(data_dir, 'LabelsArduino.csv')
fnameWrite = os.path.join(tag_dir, 'DataAug.csv')
MaxPages = 200000
MaxExamples = 280

def get_sensor_data(path, posSup, posInf): #Obtener los datos de un ejemplo a partir de su ubicacion en el csv
    f = open(path)
    sensordata = f.read()
    f.close()
    sensorlines = sensordata.split('\n')
    sensorheader = sensorlines[0].split(',')
    sensorlines = sensorlines[0:]
    sensor_float_data = np.zeros((50, len(sensorheader)))
    i = posInf
    while (i <= posSup):
        values = [float(x) for x in sensorlines[i].split(',')]
        sensor_float_data[i - posInf, :] = values
        i += 1
    return sensor_float_data

def get_rand_and_data(randomTag): #Obtener los datos de un ejemplo a partir de la direccion de su etiqueta
    posSup = (randomTag * 50) - 1
    posInf = posSup - 49
    csvsensor = get_sensor_data(fnameData, posSup, posInf)
    return csvsensor

def get_neg():
    if(randrange(2) == 1):
        neg = -1
    else:
        neg = 1
    return neg

def get_div():
    if (randrange(2) == 1):
        div = 10
    else :
        div = 100
    return div

def get_std(path):  #Obtener desviacion estandar del conjunto de datos
    f = open(path)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[0:]

    float_data = np.zeros((len(lines)-1, len(header)))
    for i, line in enumerate(lines):
        if(len(line) > 0):
            values = [float(x) for x in line.split(',')]
            float_data[i, :] = values
    mean = float_data.mean(axis=0)
    float_data -= mean
    std = float_data.std(axis=0)
    return std


f = open(fname) #Leer datos obtenidos con el sensor
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[0:]

float_data = np.zeros((len(lines), len(header)))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')]
    float_data[i, :] = values

f = open(fnameTags) #Leer datos de las etiquetas de los ejemplos
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[0:]

Tags_data = np.zeros((len(lines), len(header)))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')]
    Tags_data[i, :] = values


cont = 0
std = get_std(fnameData) #Obtener desviacion estandar de los datos
while (cont < MaxPages):    #Obtener 200000 ejemplos nuevos aplicando ruido gaussiano a un ejemplo aleatorio
    randomTag = randrange(MaxExamples)
    tag = float_data[0][cont]
    num = Tags_data[0][randomTag]
    while (num != tag):
        randomTag = randrange(MaxExamples)
        num = Tags_data[0][randomTag]
    print(cont)
    cont += 1
    csvsensor = get_rand_and_data(randomTag + 1)
    rowcont = 0
    f = open(fnameWrite, 'a+', newline='')
    writer = csv.writer(f)
    while (rowcont < 50):
        row = csvsensor[rowcont]
        col = 0
        while (col < len(csvsensor[rowcont])):
            row[col] = round(row[col] + (get_neg() * randrange(int(round(std[col],2))*100))/randrange(100,1001), 2)
            col += 1
        writer.writerow(row)
        rowcont += 1
    f.close()