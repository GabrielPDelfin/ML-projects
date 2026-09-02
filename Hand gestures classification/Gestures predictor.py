import serial
import numpy as np
from keras.models import model_from_json
import time

from collections import Counter
 
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def return_to_idle(ser):
    cont = 0
    while (cont < 50):
        ser.readline()
        cont += 1


arduino_port = "COM3" #serial port of Arduino
baud = 9600 #arduino uno runs at 9600 baud
fileName="analog-data.csv" #name of the CSV file generated

ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
getData=str(ser.readline())
mean = [-0.61536286,  0.78203286,  7.97892714,  0.05060429,  2.18662,  1.35041643,  8.82454429,  5.37788214]
vari = [16.19709967,  1.58783037,  9.19509096,  68.63964153,  929.41690028,  62.71353118,  540.15476605,  1023.22386156]
# load json and create model
json_file = open('modelDeepLSTM.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelDeepLSTM.h5")
print("Loaded model from disk")
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
threeD_data = np.zeros((1, 50, 8))
#file = open(fileName, "a")
#print("Created file")

#display the data to the terminal
q = []
read = 1
sensor_float_data = np.zeros((50, 8))
cont = 0
print('Start: ')
while (read == 1):
    if (cont == 50):
        threeD_data[0] = sensor_float_data
        out = loaded_model.predict(threeD_data)
        out = np.argmax(out,axis=1)
        if (len(q)==4):
            if (q[0] == 0 and most_frequent(q[1:]) != 0 and q.count(0) == 1):
                print('pred:')
                print(most_frequent(q[1:]))
                return_to_idle(ser)
            q.pop(0)
        q.append(out[0])
        print(out[0])
        cont -= 10
        sensor_float_data[:40] = sensor_float_data[10:10+40]
    getData=str(ser.readline())
    data=getData[2:][:-5]
    datalines = data.split('\n')
    dataheader = datalines[0].split(',')
    datalines = datalines[0:]
    try:
        values = [float(x) for x in datalines[0].split(',')]
        sensor_float_data[cont, :] = values
        sensor_float_data[cont, :] -= mean
        sensor_float_data[cont, :] /= vari
        cont += 1
    except:
        print(".")



#threeD_data[0] = sensor_float_data



#threeD_data -= mean
#threeD_data /= std
#out = loaded_model.predict(threeD_data)
#out = np.argmax(out,axis=1)
#print(out[0])




#print(sensor_float_data)
#print(datalines[0])
#print(values)
#add the data to the file
#file = open(fileName, "a") #append the data to the file
#file.write(data + "\\n") #write data with a newline

#close out the file
#file.close()