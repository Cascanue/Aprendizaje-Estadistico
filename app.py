from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar los modelos
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('modelo_kmeans.pkl')

# Ruta 1: Predicción individual
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.json
        recencia = float(datos['recencia'])
        frecuencia = float(datos['frecuencia'])
        monetario = float(datos['monetario'])

        datos_log = np.array([[np.log1p(recencia), np.log1p(frecuencia), np.log1p(monetario)]])
        datos_escalados = scaler.transform(datos_log)
        cluster_asignado = kmeans.predict(datos_escalados)[0]
        
        return jsonify({
            'exito': True,
            'cluster': int(cluster_asignado)
        })
    except Exception as e:
        return jsonify({'exito': False, 'error': str(e)})

# Ruta 2: Prueba Masiva (Lotes CSV)
@app.route('/predecir_lote', methods=['POST'])
def predecir_lote():
    try:
        datos = request.json
        clientes = datos.get('clientes', [])
        
        if not clientes:
            return jsonify({'exito': False, 'error': 'No se recibieron clientes'})

        # Extraer y transformar los datos de todos los clientes a la vez
        matriz_clientes = []
        for c in clientes:
            r = float(c['recencia'])
            f = float(c['frecuencia'])
            m = float(c['monetario'])
            matriz_clientes.append([np.log1p(r), np.log1p(f), np.log1p(m)])
            
        matriz_clientes = np.array(matriz_clientes)
        datos_escalados = scaler.transform(matriz_clientes)
        
        # Predecir todos de golpe
        predicciones = kmeans.predict(datos_escalados)
        
        # Convertir a una lista normal de Python para enviarla como JSON
        resultados = [int(p) for p in predicciones]
        
        return jsonify({
            'exito': True,
            'resultados': resultados
        })
    except Exception as e:
        return jsonify({'exito': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)