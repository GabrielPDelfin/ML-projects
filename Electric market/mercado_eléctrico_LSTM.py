import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

import seaborn as sns
import matplotlib.pyplot as plt

def descargar_dia(fecha):
    
    fecha_str = fecha.strftime("%Y%m%d")
    
    url = f"https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{fecha_str}"
    
    if(fecha_str == '20251030'):
        url += ".3"
    elif(fecha_str == '20251127'):
        url += ".2"
    else:
        url += ".1"
        
        # 2. Hacer la petición a la URL para obtener el archivo
    headers = {
        "User-Agent": "Mozilla/5.0"
    }  # Buena práctica para evitar bloqueos
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Lanza un error si la descarga falla
        
        
        # 2. Decodificar contenido
    text = response.content.decode("latin1")
    
        # 3. Separar líneas y limpiar
    lines = [l for l in text.splitlines() if l.strip()]
    
        # 4. Quitar cabecera y última línea '*'
    data_lines = []
    for l in lines:
        if l.startswith("MARGINALPDBC"):
            continue
        if l.strip() == "*":
            continue
        data_lines.append(l)
    
        # 5. Crear DataFrame desde CSV con separador ;
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=";",
        header=None,
    )
    
    df = df[df.columns[:-1]]
    
    df.columns = ["year", "month", "day", "hour", "price_pt", "price_es"]
        # 6. Crear datetime
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])#.dt.date

        # 7. Reordenar columnas
    df = df[["date", "year", "month", "day", "hour", "price_pt", "price_es"]]

    return df


def descargar_rango(fecha_inicio, fecha_fin):
    
    fecha_actual = fecha_inicio
    datos = []
    
    while fecha_actual <= fecha_fin:
        print(f"Descargando {fecha_actual.strftime('%Y%m%d')}")
        
        df = descargar_dia(fecha_actual)
        
        
        if df is not None:
            datos.append(df)
        
        fecha_actual += timedelta(days=1)
    
    return pd.concat(datos, ignore_index=True)


def crear_lags(df, window_size):

    for i in range(1, window_size + 1):
        df[f"lag_pt_{i}"] = df["price_pt"].shift(i)
        
    for i in range(1, window_size + 1):
        df[f"lag_es_{i}"] = df["price_es"].shift(i)
    
    df = df.dropna()
    return df


window_size = 24
quarter_hours = 96

fecha_inicio = datetime(2025, 10, 1)
fecha_fin = datetime(2026, 5, 26)
#fecha_inicio = datetime(2026, 5, 24)
#fecha_fin = datetime(2026, 5, 26)

datosRango = descargar_rango(fecha_inicio, fecha_fin)
datosRango = crear_lags(datosRango, window_size)


x = datosRango[[col for col in datosRango.columns if col not in ['price_pt', 'price_es', 'date']]].values

y = datosRango[["price_pt", "price_es"]].values


split = int(len(x) * 0.8)

x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]

scaler_x = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_val_scaled = scaler_x.transform(x_val)

scaler_y = StandardScaler()

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
x_val_scaled = x_val_scaled.reshape((x_val_scaled.shape[0], x_val_scaled.shape[1], 1))

model = Sequential()

model.add(LSTM(64, activation='tanh', input_shape=(x_train_scaled.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mae', metrics=["mae"])
model.summary()

early_stop = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

model.fit(x_train_scaled, y_train_scaled, epochs=10, batch_size=32, 
          validation_data=(x_val_scaled, y_val_scaled), callbacks=[early_stop])

y_pred_scaled = model.predict(x_val_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_val_original = scaler_y.inverse_transform(y_val_scaled)

mae_lstm = mean_absolute_error(y_val, y_pred)

print("-------------------MAE: ",  mae_lstm)

last_row = datosRango.iloc[-1]

next_hour = (last_row['hour'] + 1) % (quarter_hours + 1)

next_date = last_row["date"]

if next_hour == 0:
    next_hour +=1
    next_date = next_date + pd.Timedelta(days=1)


lags_pt = [datosRango["price_pt"].iloc[-i] for i in range(1, window_size + 1)]
lags_es = [datosRango["price_es"].iloc[-i] for i in range(1, window_size + 1)]

x_next = [next_date.year, next_date.month, next_date.day, next_hour] + lags_pt + lags_es

x_next_scaled = scaler_x.transform([x_next])
x_next_scaled = x_next_scaled.reshape((1, x_next_scaled.shape[1], 1))


y_next_scaled = model.predict(x_next_scaled)

y_next = scaler_y.inverse_transform(y_next_scaled)

#y_next_list.append(y_next_scaled)

value_1_pred = y_next[0][0]
value_2_pred = y_next[0][1]

x_next.insert(0, next_date)
x_next.insert(5, value_1_pred)
x_next.insert(6, value_2_pred)


new_row = pd.DataFrame([x_next], columns=datosRango.columns)

datosRango = pd.concat([datosRango, new_row], ignore_index=True)

print(datosRango.iloc[-1].head(10))

'''
y_next_list = []
for _ in range(1, window_size + 1):
    
    
    
    
    new_row = {
        "date": next_date,
        "year": next_date.year,
        "month": next_date.month,
        "day": next_date.day,
        "hour": next_hour,
        "price_pt": value_1_pred,
        "price_es": value_2_pred
    }
    
    fila_Nueva = pd.concat([pd.DataFrame([new_row]), dfNuevo], axis=0, join='outer')

    datosRango = pd.concat([datosRango, fila_Nueva], ignore_index=True)
    '''