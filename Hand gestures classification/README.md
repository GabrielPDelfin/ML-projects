Hand gesture classification project using an LSTM deep learning model and an Arduino Uno microcontroler with a gyroscope, accelerometer and a Kalman filter algorithm.
For more details see https://oa.upm.es/71348/

The sample data was first manually obtained with the Arduino Uno using the files in the 'MPU6050_Logger' directory, the labels of each gesture were saved in the 'LabelsArduino.csv' file, while the data was saved in 'DataArduino.csv'.

In order to improve the LSTM model's performance, the data was augmented by applying Gaussian noise, this data is inside the 'Data directory'.

The Jupyter notebook called 'LSTM_Model.ipynb' contains the declaration of the LSTM model, its training and validation. The final model and its weights were saved in the 'modelDeepLSTM.json' and 'modelDeepLSTM.h5' files, respectively.

The 'Gestures predictor.py' file contains the Python code that imports the saved LSTM model and uses it to continuously classify the gestures that the user performs using the Arduino module.