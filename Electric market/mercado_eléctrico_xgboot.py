import io
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

#-----------------------Descarga los datos de un día-----------------------------
def download_day(fecha):
    
    options = ['.1', '.2', '.3']
    
    date_str = fecha.strftime('%Y%m%d')
    
    url = f"https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{date_str}"
    #--------Probar a descargar con distintas urls---------------------
    for option in options:
        try:
            urlComplete = url + option
            
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }
            response = requests.get(urlComplete, headers=headers)
            response.raise_for_status()
            
            #----------Decodificar contenido-------------------------------
            text = response.content.decode('latin1')
        
            #--------------Separar líneas y limpiar-----------------------------
            lines = [l for l in text.splitlines() if l.strip()]
        
            #---------------Quitar cabecera y última línea '*'-------------------
            data_lines = []
            for l in lines:
                if l.startswith('MARGINALPDBC'):
                    continue
                if l.strip() == '*':
                    continue
                data_lines.append(l)
        
            #-------------Crear DataFrame desde CSV con separador---------------
            df = pd.read_csv(
                io.StringIO('\n'.join(data_lines)),
                sep=';',
                header=None,
                )
        
            df = df[df.columns[:-1]]
            #--------------Columnas del dataframe--------------------------------
            df.columns = ['year', 'month', 'day', 'hour', 'price_pt', 'price_es']
            
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.date

            df = df[['date', 'year', 'month', 'day', 'hour', 'price_pt', 'price_es']]
            
            return df
        #----Lanza mensaje de error si no se han conseguido descargar los datos--
        except requests.exceptions.RequestException as e:
            print(f"Failed to download from {urlComplete}: {e}")
            
    print('All download attempts failed.')
    return False

#----------Descarga los datos pertenecientes a un periodo de tiempo-------------
def download_range(start_date, end_date, folder_path):
    
    current_date = start_date
    data = []
    
    #---------------Comprobar si existe carpeta de datos--------------------
    os.makedirs(folder_path, exist_ok=True)
    
    while current_date <= end_date:
        
        #----------Comprobar si existe archivo con datos de la fecha actual-----
        file_path = os.path.join(folder_path, current_date.strftime('%Y%m%d') + '.csv')
        
        #---------------Si el archivo ya existe, leer csv----------------------
        if os.path.exists(file_path):
            print(f"Reading {current_date.strftime('%Y%m%d')}")
            df = pd.read_csv(file_path)
        #--------Si el archivo no existe, descargar y guardar como un csv-------
        else:
            print(f"Downloading {current_date.strftime('%Y%m%d')}")
            df = download_day(current_date)
            df.to_csv(file_path, index=False)
        
        
        if df is not None:
            data.append(df)
        #---------------------Actualizar fecha---------------------------------
        current_date += timedelta(days=1)
    
    return pd.concat(data, ignore_index=True)

#------------------Crea los lags de los precios de Portugal y España------------
def create_lags(df, window_size):

    dataPT = {}
    
    for i in range(1, window_size + 1):
        dataPT[f"lag_pt_{i}"] = df['price_pt'].shift(i)
        
    
    df = df.assign(**dataPT)
    
    dataES = {}
    
    for i in range(1, window_size + 1):
        dataES[f"lag_es_{i}"] = df['price_es'].shift(i)
    
    #-------------Insertar lags en el dataframe y borrar filas con nulos
    df = df.assign(**dataES)
    
    df = df.dropna()
    return df


#-----------Ruta donde se guardan en csv los datos de cada día-----------------
folder_path = './Datos'
#----------Tamaño de ventana que se usa para calcular los lags-----------------
window_size = 96

#Los datos contienen una entrada por cada cuarto de hora. En un día hay 24*4 cuartos de hora
quarter_hours = 96


#----------------------Obtener datos para el entrenamiento--------------------
'''
***********NOTA: Desde el 1 de octubre del 2025 en adelante 
los datos contienen una entrada por cada cuarto de hora. 
Antes de esa fecha los datos contienen una entrada por cada hora
'''
start_date = datetime(2025, 10, 1)
end_date = datetime.now()

dataRange = download_range(start_date, end_date, folder_path)

dataRange = create_lags(dataRange, window_size)

#---------------------------Preparar features y los targets---------------------
x = dataRange[[col for col in dataRange.columns if col not in ['price_pt', 'price_es', 'date']]].values

y = dataRange[['price_pt', 'price_es']].values

#---------------------Separar datos de entrenamiento y pruebas-------------------
split = int(len(x) * 0.8)

x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]


#---------------------------------Crear, entrenar y validar modelo--------------
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
)

model.fit(x_train, y_train)

y_pred = model.predict(x_val)

print('MAE:', mean_absolute_error(y_val, y_pred))

#-Predecir los precios del día siguiente (los 96 cuartos de hora siguientes)----
for _ in range(1, quarter_hours + 1):
    #-----Calcular fecha a partir de la última fila del conjunto de datos-------
    last_row = dataRange.iloc[-1]

    next_hour = (last_row['hour'] + 1) % (quarter_hours + 1)

    next_date = pd.to_datetime(last_row['date'])

    if next_hour == 0:
        next_hour +=1
        next_date = next_date + pd.Timedelta(days=1)

    #-------------------Calcular lags de los precios a predecir-----------------
    lags_pt = [dataRange['price_pt'].iloc[-i] for i in range(1, window_size + 1)]
    lags_es = [dataRange['price_es'].iloc[-i] for i in range(1, window_size + 1)]

    #-----------------------Preparar features--------------------------------
    x_next = [next_date.year, next_date.month, next_date.day, next_hour] + lags_pt + lags_es

    #------------------------Predecir precios--------------------------------
    y_next = model.predict([x_next])


    value_1_pred = y_next[0][0]
    value_2_pred = y_next[0][1]

    #---------Crear fila nueva de datos e insertar al final del conjunto de datos
    x_next.insert(0, next_date.strftime('%Y-%m-%d'))
    x_next.insert(5, value_1_pred)
    x_next.insert(6, value_2_pred)

    new_row = pd.DataFrame([x_next], columns=dataRange.columns)

    dataRange = pd.concat([dataRange, new_row], ignore_index=True)
    
#--------Crear gráfico de barras con los precios del día siguiente predichos----

plt.figure(figsize=(15,5))

x = np.arange(len(dataRange['hour'].tail(quarter_hours)))
width = 0.3

plt.bar(x + width/2, dataRange['price_es'].tail(quarter_hours), width, label='Spanish prices', color='g')
plt.bar(x - width/2, dataRange['price_pt'].tail(quarter_hours), width, label='Portugese prices', color='#00CED1')

plt.xlabel('Quarter of an hour')
plt.ylabel('Price (EUR/MWh)')
plt.title(f"Predicted prices for {dataRange.iloc[-1]['date']}")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))

plt.plot(range(1, (len(x_val[:, 3])) + 1), y_pred[:, 0], label='Predicted portugese prices', color='#00CED1')
plt.plot(range(1, (len(x_val[:, 3])) + 1), y_val[:, 0],  label='Real portugese prices', linestyle = 'dotted', color ='r')

plt.xlabel('Quarter of an hour')
plt.ylabel('Price (EUR/MWh)')
plt.title("Predicted prices vs real prices (Portugal)")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))

plt.plot(range(1, (len(x_val[:, 3])) + 1), y_pred[:, 1], label='Predicted spanish prices', color='g')
plt.plot(range(1, (len(x_val[:, 3])) + 1), y_val[:, 1],  label='Real spanish prices', linestyle = 'dotted', color = 'r')


plt.xlabel('Quarter of an hour')
plt.ylabel('Price (EUR/MWh)')
plt.title("Predicted prices vs real prices (Spain)")

plt.legend()
plt.tight_layout()
plt.show()

csvDate = dataRange.iloc[-1]['date'].replace('-', '')
dataRange.tail(quarter_hours).to_csv(folder_path + '/' + csvDate + '_prediction.csv', index=False)



