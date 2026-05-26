# Caso de Estudio: FoodVision AI

La cadena de restaurantes FoodVision Colombia posee múltiples sedes en diferentes ciudades del país y durante el último año ha comenzado a presentar diversos problemas operativos y administrativos:

- Pérdida de alimentos por mala planificación.
- Largas filas en horas pico.
- Pedidos entregados incorrectamente.
- Baja satisfacción de clientes.
- Problemas para predecir demanda diaria.
- Incremento de costos operativos.
- Mala distribución del personal de cocina.
- Dificultades para identificar clientes frecuentes.
- Retrasos en domicilios.

La gerencia desea implementar un sistema basado en Inteligencia Artificial, Machine Learning y Deep Learning que permita automatizar decisiones y mejorar el rendimiento del restaurante.

Para ello, el departamento de tecnología entrega un archivo llamado:

- restaurante_foodvision.xlsx

El dataset contiene información histórica de ventas y operaciones del restaurante.

## Variables del Dataset

- id_pedido: Identificador del pedido
- ciudad: Ciudad
- tipo_comida: Hamburguesa, pizza, sushi, etc.
- hora_pedido: Hora del pedido
- cantidad_productos: Número de productos
- valor_total: Valor del pedido
- tiempo_preparacion: Minutos de preparación
- metodo_pago: Efectivo, tarjeta, app
- cliente_frecuente: Sí / No
- calificacion_cliente: 1 a 5
- clima: Soleado, lluvia, tormenta
- demora_entrega: Sí / No
- satisfaccion_cliente: Alta / Media / Baja
- ventas_dia: Total de ventas diarias

## Objetivo General

Diseñar una solución inteligente capaz de:

- Predecir retrasos en entregas.
- Analizar satisfacción del cliente.
- Identificar patrones de consumo.
- Predecir ventas futuras.
- Optimizar tiempos de atención.
- Detectar riesgos operativos.
- Comparar modelos de ML y Deep Learning.

## Parte 1: Identificación del Problema

Preguntas analíticas:

- ¿Cuáles son los principales problemas operativos del restaurante?
- ¿Qué problemas podrían resolverse usando IA?
- ¿Qué tareas podrían automatizarse?
- ¿Qué consecuencias genera una mala predicción de demanda?
- ¿Qué variables podrían afectar la satisfacción del cliente?
- ¿Qué riesgos tendría confiar totalmente en IA dentro del restaurante?
- ¿Qué problemas podrían surgir si existen errores en el dataset?
- ¿Por qué el clima podría influir en las ventas?
- ¿Qué impacto tiene el tiempo de preparación sobre el negocio?
- ¿Cómo podría ayudar el Deep Learning en este contexto?

## Parte 2: Identificación del Tipo de Problema

Clasifiquen cada situación:

- Predecir satisfacción del cliente: ?
- Predecir ventas diarias: ?
- Agrupar clientes según hábitos: ?
- Detectar retrasos de entrega: ?
- Recomendar productos: ?

## Parte 3: Limpieza y Calidad de Datos

El dataset contiene errores intencionales:

- Inconsistencias en ciudad ("Medellín", "medellin", "MEDELLIN")
- Valores negativos (tiempo_preparacion, valor_total)
- Valores faltantes (NaN)
- Outliers (tiempo_preparacion muy alto)

Actividades:

- Detectar valores inválidos.
- Corregir inconsistencias.
- Eliminar registros duplicados.
- Imputar valores faltantes.
- Validar rangos lógicos.
- Estandarizar nombres de ciudades.
- Explicar cómo afecta la mala calidad de datos al modelo.

## Parte 4: Transformación de Datos

Actividades:

- Escalado: Min-Max Scaling, Z-score
- Codificación: One-Hot Encoding, Label Encoding
- Feature Engineering: crear nuevas variables (categoria_consumo, nivel_demora, cliente_premium, promedio_compra_cliente)
- Reducción de Dimensionalidad: PCA, Selección de atributos

## Parte 5: Desarrollo de Modelos de Machine Learning

Seleccionar y justificar al menos:

- Árbol de decisión
- Random Forest
- Regresión logística
- KNN

Actividades técnicas:

- Separar variables predictoras y objetivo.
- Dividir dataset en entrenamiento y prueba.
- Entrenar modelos.
- Realizar predicciones.
- Comparar resultados.

## Parte 6: Evaluación del Modelo

Para clasificación:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión

Para regresión:

- MAE
- MSE
- RMSE
- R²

## Parte 7: Deep Learning

Comparar Machine Learning tradicional contra redes neuronales profundas.

- Red neuronal básica (1 capa oculta)
- Red neuronal profunda (varias capas ocultas)

Comparar:

- Accuracy
- Loss
- Tiempo de entrenamiento
- Rendimiento general
- Capacidad de generalización

## Parte 8: Preguntas Analíticas y Críticas

Análisis técnico:

- ¿Por qué un modelo podría tener buen accuracy en entrenamiento y mal rendimiento en prueba?
- ¿Qué significa overfitting?
- ¿Por qué demasiadas epochs pueden afectar el modelo?
- ¿Qué función cumple batch_size?
- ¿Por qué la normalización mejora algunos algoritmos?
- ¿Qué riesgos existen si el dataset está desbalanceado?
- ¿Por qué accuracy no siempre representa un buen modelo?
- ¿Qué consecuencias tendría una predicción errónea en entregas?
- ¿Por qué Deep Learning requiere mayor capacidad computacional?
- ¿Qué ventajas ofrece Random Forest frente a un árbol simple?

Análisis crítico y ético:

- ¿Es correcto que una IA determine automáticamente prioridades de pedidos?
- ¿Qué riesgos existen si el sistema falla en horas pico?
- ¿Cómo evitar sesgos en recomendaciones automáticas?
- ¿Qué consecuencias tendría depender totalmente de IA?
- ¿Quién sería responsable por decisiones incorrectas del sistema?

## Parte 9: Desarrollo del Aplicativo Python

El aplicativo debe permitir:

- Leer archivos Excel.
- Mostrar dataset original.
- Limpiar datos automáticamente.
- Transformar variables.
- Entrenar modelos ML.
- Entrenar redes neuronales.
- Comparar modelos.
- Mostrar métricas.
- Generar gráficas.
- Realizar predicciones.

## Parte 10: Entregables

- Documento técnico: definición del problema, justificación del modelo, explicación del dataset, evidencia de limpieza, resultados, interpretación crítica, conclusiones.
- Código Python: comentarios claros, buenas prácticas, separación modular, explicación de funciones.
- Presentación final: problema, arquitectura, comparación, riesgos y limitaciones, recomendaciones.

## Condiciones especiales

- No se permite copiar código completo generado automáticamente sin explicación.
- Todas las decisiones técnicas deben justificarse.
- Deben explicar cada transformación aplicada.
- Deben argumentar por qué un modelo fue mejor.
- Deben identificar posibles errores del sistema.
