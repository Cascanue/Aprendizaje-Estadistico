import pandas as pd
import numpy as np
import joblib

print("Cargando datos brutos...")
df = pd.read_csv('data.csv', encoding='unicode_escape')
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df['TotalGasto'] = df['Quantity'] * df['UnitPrice']

print("Calculando RFM...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
fecha_actual = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (fecha_actual - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalGasto': 'sum'
}).reset_index()

# 1. Mantenemos los nombres en inglés temporalmente para que el modelo no se queje
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("Aplicando modelo de Inteligencia Artificial...")
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('modelo_kmeans.pkl')

rfm_log = rfm.copy()
rfm_log['Recency'] = np.log1p(rfm['Recency'])
rfm_log['Frequency'] = np.log1p(rfm['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm['Monetary'])

# 2. Transformamos y predecimos usando los nombres que el modelo ya conoce
datos_escalados = scaler.transform(rfm_log[['Recency', 'Frequency', 'Monetary']])
rfm['cluster'] = kmeans.predict(datos_escalados)

# 3. Ahora sí, traducimos las columnas al español para nuestro Dashboard
rfm = rfm.rename(columns={'Recency': 'recencia', 'Frequency': 'frecuencia', 'Monetary': 'monetario'})

# Guardar solo las columnas necesarias para el gráfico web
archivo_salida = 'dataset_visualizacion.csv'
rfm[['recencia', 'frecuencia', 'monetario', 'cluster']].to_csv(archivo_salida, index=False)
print(f"¡Listo! Archivo '{archivo_salida}' generado para tu Dashboard.")