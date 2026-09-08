import csv as csv
import math
import cv2
from pylab import *
#from matplotlib import pyplot as plt

def main():
    sum = 0
    nodo1 = []
    nodo2 = []

    #plot([1,2,3],[1,2,3])
    #show()
    imagen = cv2.imread("nodo100001.jpg")
    #plt.imshow(imagen)
    #imshow(imagen)
    #show()

    with open("histogramas/nodo1.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            for column in row:
                nodo1.append(column)
    with open("histogramas/nodo2.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            for column in row:
                nodo2.append(column)

    #print(nodo2[256])

    enum = 0

    for valor1 in nodo1:
        if enum > 255:
            sum += pow(nodo1[enum] - nodo2[enum],2)
        enum += 1
    sum = math.sqrt(sum)

    if sum < 54000:
        print('nodo1 ')
    print(sum)

main()