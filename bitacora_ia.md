# Bitácora de Uso de Inteligencia Artificial - Entrega T1

En cumplimiento de las políticas del curso, se detalla el uso de herramientas de IA generativa para el desarrollo de la primera entrega.

## 1. Registro de Interacciones

| Fecha | Herramienta | Objetivo del Prompt | Resultado / Validación |
| :--- | :--- | :--- | :--- |
| 07/07/2026 | IA Generativa | Revisión de estructura del README y validación técnica del informe inicial T1 de concreto. | Se confirmó el cumplimiento de la rúbrica y se detectó un enlace roto en el README. |
| 07/07/2026 | IA Generativa | Redacción sintética y compacta del enfoque del Marco VDS (PCS Framework) para el repositorio. | Se obtuvo un texto directo adaptado a las 1030 filas y variables del dataset. |
| 07/07/2026 | IA Generativa | Diagnóstico de error de renderizado (bloque de código de la estructura de carpetas se mostraba como texto corrido en la vista previa de GitHub). | Se identificó como posible causa un problema de codificación de saltos de línea (CRLF) o cierre incorrecto de comillas triples. |
| 13/07/2026 | IA Generativa | Traducción al español del archivo README del dataset (Concrete Compressive Strength - UCI), manteniendo estructura y formato original. | Se generó la traducción completa conservando el formato de secciones, separadores y variables; los títulos de las publicaciones científicas se dejaron en su idioma original por convención de citación académica. |
| 14/07/2026 | IA Generativa | Diagnóstico y solución del error ModuleNotFoundError: No module named 'pandas' al ejecutar el script 01_eda.ipynb | Se identificó la falta de instalación de las librerías en el entorno activo. Se guio en el uso del comando %pip dentro del notebook y el comando pip en PowerShell para resolver el conflicto de manera exitosa. |
| 14/07/2026 | IA Generativa | Diagnóstico de error al cargar el archivo `Concrete_Data.csv` real del equipo: pandas leía todas las columnas como una sola. | Se identificó que el archivo usa `;` como separador (formato original de UCI) en vez de `,`. Corregido con `sep=";"` en `pd.read_csv`. |
| 14/07/2026 | IA Generativa | Diagnóstico de `FileNotFoundError` al guardar figuras (`plt.savefig`) en una carpeta `docs/figures/` inexistente. | Se agregó `os.makedirs("../docs/figures", exist_ok=True)` a la celda de imports del notebook. Verificado ejecutando el notebook completo sin crear la carpeta de antemano, confirmando que se genera automáticamente. |
| 18/07/2026 | IA Generativa | Generación del código fuente final modular (`src/data_loader.py`, `train.py`, `evaluate.py`, `interpretability.py`, `main.py`) , incluyendo el análisis SHAP por rango de resistencia (baja/media/alta). | El pipeline completo fue **ejecutado de extremo a extremo** (`python main.py`) para validar que corre sin errores antes de subirlo al repositorio. Se verificó además la estabilidad del modelo con y sin las 25 filas duplicadas (R²=0.917 vs. 0.937), conforme al marco VDS. |
| 21/07/2026 | IA Generativa | Diagnóstico de dos errores al ejecutar `main.py` en la computadora de un integrante: `FileNotFoundError` del CSV y `requirements.txt` no encontrado. | Se identificó que el integrante ejecutaba los comandos desde la raíz del repositorio en vez de la carpeta `src/`, lo cual rompe las rutas relativas (`../data/...`) del código. Se identificó además que el archivo `requirements.txt` local tenía 0 bytes (vacío), pese a existir. El equipo verificó ambos hallazgos con `dir` antes de aplicar la corrección. |


## 2. Declaración de Co-Creación y Validación
La IA fue utilizada exclusivamente como un asistente de revisión estructural, ortográfica y de síntesis metodológica. Toda la contextualización del problema en el entorno constructivo peruano (regiones de Cusco/Apurímac), las normativas técnicas aplicadas (NTP 339.034 / ASTM C39) y la selección del dataset de la UCI fueron analizadas, validadas y redactadas bajo el criterio técnico de los integrantes del equipo.

Todo el contenido técnico generado con apoyo de IA fue revisado por el equipo antes de su entrega, verificando exactitud normativa y coherencia con fuentes primarias.
