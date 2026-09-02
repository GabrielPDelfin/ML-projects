import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

import shap

#------------------Leer fichero arff------------------
data, meta = arff.loadarff('./credit_fraud.arff')

df = pd.DataFrame(data)

#------------------Convertir columnas con valores de tipo object a tipo string------------------
for column in df.columns:
    if(df[column].dtypes == 'object'):
        df[column] = df[column].str.decode("utf-8")

#------------------Codificar variables categóricas a números enteros------------------
df['over_draft'] = df['over_draft'].map({ '<0':1, '0<=X<200':2, '>=200':3, 'no checking':4}).astype(int)

df['credit_history'] = df['credit_history'].map({ 'no credits/all paid':1, 'all paid':2, 'existing paid':3, 'delayed previously':4, 'critical/other existing credit':5}).astype(int)

df['purpose'] = df['purpose'].map({ 'new car':1, 'used car':2, 'furniture/equipment':3, 'radio/tv':4, 'domestic appliance':5, 'repairs':6, 'education':7, 'vacation':8, 'retraining':9, 'business':10, 'other':11}).astype(int)

df['Average_Credit_Balance'] = df['Average_Credit_Balance'].map({ '<100':1, '100<=X<500':2, '500<=X<1000':3, '>=1000':4, 'no known savings':5}).astype(int)

df['employment'] = df['employment'].map({ 'unemployed':1, '<1':2, '1<=X<4':3, '4<=X<7':4, '>=7':5}).astype(int)

df['personal_status'] = df['personal_status'].map({ 'male div/sep':1, 'female div/dep/mar':2, 'male single':3, 'male mar/wid':4, 'female single':5}).astype(int)

df['other_parties'] = df['other_parties'].map({ 'none':1, 'co applicant':2, 'guarantor':3}).astype(int)

df['property_magnitude'] = df['property_magnitude'].map({ 'real estate':1, 'life insurance':2, 'car':3, 'no known property':4}).astype(int)

df['other_payment_plans'] = df['other_payment_plans'].map({ 'bank':1, 'stores':2, 'none':3}).astype(int)

df['housing'] = df['housing'].map({ 'rent':1, 'own':2, 'for free':3}).astype(int)

df['job'] = df['job'].map({ 'unemp/unskilled non res':1, 'unskilled resident':2, 'skilled':3, 'high qualif/self emp/mgmt':4}).astype(int)

df['own_telephone'] = df['own_telephone'].map({ 'none':1, 'yes':2}).astype(int)

df['foreign_worker'] = df['foreign_worker'].map({ 'yes':1, 'no':2}).astype(int)

#------------------El modelo XGBoost necesita que las etiquetas sean binarias------------------
df['class'] = df['class'].map({ 'good':0, 'bad':1}).astype(int)

#------------------Preparar datos de entrenamiento y de pruebas------------------
x = df.loc[:, 'over_draft':'foreign_worker']
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#------------------Crear el modelo XGBoot------------------
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.5,
    max_depth=2,
    random_state=42,
    eval_metric="logloss"
)

#------------------Entrenar el modelo------------------
model.fit(x_train, y_train)

#------------------Hacer predicciones------------------
y_pred = model.predict(x_test)

#------------------Obtener la precisión------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

#------------------Cuantificar la importancia de cada atributo al hacer predicciones usando SHAP------------------
explainer = shap.Explainer(model, x_train)
shap_values = explainer(x_test)

importance = np.abs(shap_values.values).mean(axis=0)

feature_importance = pd.Series(importance, index=x.columns).sort_values(ascending=False)

#------------------Crear gráfico de barras con la importancia de los atributos------------------
graph = feature_importance.plot(kind="barh", figsize=(10, 10))

for i, v in enumerate(feature_importance.values):
    graph.text(v + 0.0, i, str(round(v,2)), va='center')

plt.xlabel("Importance")
plt.ylabel("Attributes")

plt.show()
