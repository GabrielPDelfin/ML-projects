import os
import numpy as np
import csv

data_dir = 'C:/Users/penad/Desktop/MUIA/TFM/Logger/Data'
fname = os.path.join(data_dir, 'LabelsArduino.csv')

#cont = 0
#from random import randrange
#while (cont <= 1000):
#    if (randrange(2) == 1):
#        div = 100
#    else :
#        div = 100
#    print( randrange(101)/div)
#    cont += 1

randnums= np.random.randint(0,7,200000)
f = open(fname, 'a+', newline='')
writer = csv.writer(f)
writer.writerow(randnums)
f.close()