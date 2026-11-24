# Credit-Risk-XAI
Explainable machine learning for credit risk assessment and loan default prediction. 

Instrucciones: https://docs.google.com/document/d/11morUR53q34YpBN1F5M6iHKRaeEhGFQU/edit?usp=sharing&ouid=102283272886650531883&rtpof=true&sd=true

Dataset: https://www.kaggle.com/datasets/ajay1735/hmeq-data

## Limpieza y análisis exploratorio

Ejecutar `01-loadTransform.ipynb` en orden con `notebooks` como directorio de trabajo del kernel, con pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, IPython, ipykernel y fastparquet.

El proyecto utiliza solo rutas relativas en cadenas, como `../input/hmeq.csv` y `../outputs/`, sin utilizar `pathlib` ni `os`. PowerShell debe estar disponible en Windows para crear las carpetas de salida desde el notebook.

La limpieza aplica dos filtros antes del EDA y de las imputaciones sobre `VALUE`, `CLAGE`, `NINQ`, `CLNO`, `YOJ`, `DEROG`, `DELINQ` y `DEBTINC`. Primero elimina 4 filas con las ocho variables nulas. Después elimina 279 filas adicionales con 4 o más nulos entre esas mismas variables. En total descarta 283 de 5960 filas (4,7483 %) y conserva 5677. `DEBTINC` conserva sus nulos para análisis y queda fuera de K-means, al igual que su indicador de ausencia. Después de los filtros no quedan nulos en `CLNO`; su política de imputación con cero sigue documentada.

La segmentación compara k de 2 a 10 mediante una tabla de silueta e incluye una gráfica del método del codo para k de 1 a 10 y un diagrama de siluetas individuales del modelo elegido. La línea vertical de la curva de inercia marca la elección por silueta; no estima automáticamente un codo. La ejecución revisada selecciona k=4 y no utiliza PCA. Los grupos son exploratorios y sus etiquetas no representan niveles de riesgo.

- `outputs/hmeq_limpio.parquet`: datos limpios y banderas de ausencia; `DEBTINC` mantiene nulos.
- `outputs/hmeq_con_grupos.parquet`: asignaciones, `ID_FILA`, `GRUPO` y `SILUETA`.
- `outputs/grupos_kmeans_k4/`: cuatro parquets, con 2575, 2137, 363 y 602 solicitudes.
- `outputs/kmeans_codo.png` y `outputs/kmeans_silueta.png`: las dos gráficas incluidas en el notebook y las guías.
- [Guía completa de programación y estadística](explicacion_clean_eda.html): fundamentos, recorrido del código, ejemplos y ejercicios resueltos.
- [Guía en PDF](explicacion_clean_eda.pdf): el mismo contenido con paleta Bosque y tipografía Georgia.

El notebook describe la muestra completa. Para evaluar un clasificador, separar entrenamiento y prueba antes de ajustar imputaciones, escalado o segmentación.
