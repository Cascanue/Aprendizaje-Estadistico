# Sistema de Segmentación de Clientes Retail (Aprendizaje Estadístico) 📊

Este proyecto implementa un modelo de Machine Learning (K-Means) para segmentar automáticamente a los clientes de una empresa del sector retail basándose en su comportamiento transaccional utilizando la metodología RFM (Recencia, Frecuencia, Monetario).

##  Arquitectura del Sistema

El proyecto está diseñado bajo una arquitectura de microservicios:
- **Frontend (Vercel):** Interfaz gráfica (SPA) desarrollada con HTML5, Tailwind CSS y Vanilla JavaScript. Utiliza Chart.js para la visualización de los clústeres.
- **Backend (Render):** API REST desarrollada en Python con Flask. Carga los modelos entrenados mediante `joblib` para ejecutar inferencias en milisegundos.
- **Modelo ML:** Scikit-Learn (K-Means, StandardScaler).

## Evaluación del Modelo

Las pruebas de estabilidad geométrica arrojaron las siguientes métricas para el conjunto de datos de entrenamiento:
* **Número óptimo de clústeres (K):** 2 (Determinado por el Método del Codo).
* **Coeficiente de Silhouette:** 0.4356 (Demuestra ausencia de sobreajuste).
* **Índice de Davies-Bouldin:** 0.8995 (Confirmando una buena separación de grupos).
* **Índice de Dunn:** 0.4788

## Funcionalidades Principales

1. **Predicción Individual:** Formulario para ingresar variables de un cliente y obtener su clúster asignado con una recomendación de estrategia CRM en tiempo real.
2. **Prueba por Lotes (Batch Testing):** Carga de archivos CSV para procesar cientos de clientes simultáneamente, generando una tabla de resultados y un gráfico circular de distribución.
3. **Visualización Dinámica:** Carga de un dataset real (`dataset_visualizacion.csv`) para mapear la totalidad de la cartera de clientes en un gráfico de dispersión bidimensional.

## Enlaces del Proyecto
* **Interfaz de Usuario (Frontend):** https://aprendizaje-estadistico.vercel.app

## Autores
* Arreátegui Gutiérrez, Fabricio André
* Gomez Rivera, Diego Alfredo
* Lezama Pasco, Diegoandree Ronald
* Morgan Martinet Sebastián
* Tarrillo Villanueva, Jerson David
* Vasquez Marquina, Yair Asael 
