# Predicción de la Resistencia a Compresión del Concreto mediante Machine Learning

*Entrega T1: Definición del Problema, Datos y Repositorio*

**Curso:** Aplicaciones de IA en Estructuras

**Docente:** Ing. Kurt Soncco Sinchi

**Julio 2026**

## Descripción del Proyecto

Este proyecto busca aplicar modelos de Machine Learning (Random Forest, con comparación posterior contra Gradient Boosting/XGBoost) para predecir la resistencia a compresión del concreto a partir de las proporciones de su mezcla y edad de curado, reduciendo la dependencia del ensayo destructivo tradicional a 28 días.

Dataset: Concrete Compressive Strength Dataset (UCI)


---

## Integrantes del Equipo

| Apellidos           | Nombres           | Usuario  |
|---------------------|--------------------|----------|
| Clavijo Zavaleta    | Roberto Daniel     | rclavijo100   |
| Llamacponcca Velarde| Elizabeth Cristina | XXXXXX   |
| Sanchez Viguria     | Hans Cristian      | HansSanchezViguria   |
| Crisostomo Ramos    | Juan Felipe        | JFCRISOSTOMO   |

---

## Estructura del Repositorio
```text
├── data/
│   ├── raw/                # Dataset original sin modificar (concrete_data.csv)
│   └── processed/          # Datos limpios, listos para modelar (T2)
├── notebooks/
│   ├── 01_eda.ipynb        # Análisis exploratorio (T2)
│   ├── 02_modeling.ipynb   # Entrenamiento de modelos (T3)
│   └── 03_evaluation.ipynb # Evaluación y comparación de modelos (T3)
├── src/
│   ├── data_loader.py      # Funciones de carga y limpieza de datos
│   ├── train.py            # Script de entrenamiento
│   └── evaluate.py         # Métricas y visualizaciones de resultados
├── docs/
│   ├── T1_Informe-Inicial.docx
│   └── informe_final_IEEE.pdf # T3
├── bitacora_ia.md
├── requirements.txt
└── README.md
```

## Problema de Ingeniería

La resistencia a compresión del concreto es la propiedad mecánica más determinante para el diseño de elementos estructurales, pero su verificación exige ensayos destructivos normalizados (NTP 339.034 / ASTM C39) que toman hasta 28 días. Este proyecto busca predecir dicha resistencia a partir de la dosificación de la mezcla, permitiendo anticipar el desempeño del concreto antes del vaciado.

## Dataset


Fuente: UCI Machine Learning Repository
Observaciones: 1030 registros
Variables: 8 predictoras (cemento, escoria de alto horno, cenizas volantes, agua, superplastificante, agregado grueso, agregado fino, edad) + 1 variable objetivo (resistencia a compresión en MPa)
Tipo de datos: Numérico continuo, sin valores categóricos ni nulos


## Marco VDS (Veridical Data Science — PCS Framework)

Este proyecto sigue los principios de Predictibilidad, Computabilidad y Estabilidad propuestos por Yu & Barter (2024).

* **Predictivilidad:** Evaluaremos la precisión de los modelos mediante métricas de regresión usando una partición de datos de entrenamiento y prueba.
* **Computabilidad:** El dataset de 1030 registros es ligero, garantizando que el procesamiento y entrenamiento de los algoritmos se ejecuten eficientemente en segundos en cualquier CPU estándar.
* **Estabilidad:** Validaremos que las predicciones de resistencia a la compresión sean robustas ante variaciones aleatorias en los datos o cambios en los hiperparámetros.

Ver detalle completo en [docs/T1_Informe-Inicial.docx](docs/T1_Informe_Inicial.docx)


## Uso de IA (Bitácora)

El equipo utiliza herramientas de IA generativa (Claude, ChatGPT, GitHub Copilot) como asistentes de programación y redacción, siguiendo la política de transparencia del curso. El registro detallado de prompts, iteraciones y validaciones se documenta en docs/bitacora_ia.md.

## Referencias


Yu, B., & Barter, R. (2024). Veridical Data Science. MIT Press. https://vdsbook.com/

Yeh, I-C. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository.

Anthropic. (2026). Claude (versión Sonnet 4.5) [Modelo de lenguaje de gran escala]. https://claude.ai
