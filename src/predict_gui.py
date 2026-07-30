"""
predict_gui.py

Aplicación de escritorio (Tkinter) para predecir la resistencia a compresión
del concreto a partir de la dosificación de la mezcla, usando el modelo
XGBoost ya entrenado en T3 (main.py).

Requisito previo: haber corrido `python main.py` al menos una vez, para que
exista el archivo ../results/xgboost_model.joblib

Uso:
    cd src
    python predict_gui.py

Proyecto: Predicción de la Resistencia a Compresión del Concreto mediante ML
Curso: Aplicaciones de IA en Estructuras
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd

MODEL_PATH_XGB = "../results/xgboost_model.joblib"
MODEL_PATH_RF = "../results/random_forest_model.joblib"

# Nombre interno (usado por el modelo) -> (etiqueta en pantalla, unidad, valor por defecto)
FIELDS = [
    ("Cement",            "Cemento",                "kg/m³", 300.0),
    ("BlastFurnaceSlag",  "Escoria de alto horno",  "kg/m³", 0.0),
    ("FlyAsh",            "Cenizas volantes",       "kg/m³", 0.0),
    ("Water",             "Agua",                   "kg/m³", 180.0),
    ("Superplasticizer",  "Superplastificante",     "kg/m³", 6.0),
    ("CoarseAggregate",   "Agregado grueso",        "kg/m³", 1000.0),
    ("FineAggregate",     "Agregado fino",          "kg/m³", 780.0),
    ("Age",               "Edad de curado",         "días",  28.0),
]

# Rangos observados en el dataset de entrenamiento (para advertir extrapolación)
VALID_RANGES = {
    "Cement": (102.0, 540.0),
    "BlastFurnaceSlag": (0.0, 359.4),
    "FlyAsh": (0.0, 200.1),
    "Water": (121.8, 247.0),
    "Superplasticizer": (0.0, 32.2),
    "CoarseAggregate": (801.0, 1145.0),
    "FineAggregate": (594.0, 992.6),
    "Age": (1.0, 365.0),
}


class ConcreteStrengthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Resistencia a Compresión del Concreto")
        self.root.geometry("480x640")
        self.root.resizable(False, False)

        self.entries = {}
        self.model_choice = tk.StringVar(value="XGBoost (recomendado)")

        self._build_ui()
        self._load_models()

    # ------------------------------------------------------------------
    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Predicción de Resistencia a Compresión\ndel Concreto",
            font=("Segoe UI", 14, "bold"),
            pady=10,
        )
        title.pack()

        subtitle = tk.Label(
            self.root,
            text="Ingrese la dosificación de la mezcla (por m³)",
            font=("Segoe UI", 9, "italic"),
            fg="gray30",
        )
        subtitle.pack(pady=(0, 10))

        form_frame = tk.Frame(self.root, padx=20)
        form_frame.pack(fill="x")

        for internal_name, label_text, unit, default in FIELDS:
            row = tk.Frame(form_frame, pady=4)
            row.pack(fill="x")

            lbl = tk.Label(row, text=f"{label_text}:", width=22, anchor="w", font=("Segoe UI", 9))
            lbl.pack(side="left")

            entry = tk.Entry(row, width=10, justify="right")
            entry.insert(0, str(default))
            entry.pack(side="left", padx=5)

            unit_lbl = tk.Label(row, text=unit, width=6, anchor="w", fg="gray40", font=("Segoe UI", 9))
            unit_lbl.pack(side="left")

            self.entries[internal_name] = entry

        # Selector de modelo
        model_frame = tk.Frame(self.root, pady=10)
        model_frame.pack(fill="x", padx=20)
        tk.Label(model_frame, text="Modelo:", font=("Segoe UI", 9, "bold")).pack(side="left")
        model_selector = ttk.Combobox(
            model_frame,
            textvariable=self.model_choice,
            values=["XGBoost (recomendado)", "Random Forest"],
            state="readonly",
            width=22,
        )
        model_selector.pack(side="left", padx=10)

        # Botón de predicción
        predict_btn = tk.Button(
            self.root,
            text="Predecir Resistencia",
            command=self._predict,
            bg="#2E7D32",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=8,
            cursor="hand2",
        )
        predict_btn.pack(pady=18)

        # Resultado
        self.result_label = tk.Label(
            self.root,
            text="Resistencia predicha: — MPa",
            font=("Segoe UI", 16, "bold"),
            fg="#1B5E20",
        )
        self.result_label.pack(pady=(0, 5))

        self.warning_label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 8),
            fg="#B71C1C",
            wraplength=420,
            justify="center",
        )
        self.warning_label.pack()

        footer = tk.Label(
            self.root,
            text="Modelo entrenado sobre el dataset Concrete Compressive Strength (UCI).\n"
                 "Herramienta de apoyo — no reemplaza el ensayo normativo NTP 339.034 / ASTM C39.",
            font=("Segoe UI", 7),
            fg="gray50",
            justify="center",
        )
        footer.pack(side="bottom", pady=10)

    # ------------------------------------------------------------------
    def _load_models(self):
        self.model_xgb = None
        self.model_rf = None

        if os.path.exists(MODEL_PATH_XGB):
            self.model_xgb = joblib.load(MODEL_PATH_XGB)
        if os.path.exists(MODEL_PATH_RF):
            self.model_rf = joblib.load(MODEL_PATH_RF)

        if self.model_xgb is None and self.model_rf is None:
            messagebox.showerror(
                "Modelos no encontrados",
                "No se encontraron modelos entrenados en ../results/.\n\n"
                "Antes de usar esta ventana, corre:\n\n"
                "    cd src\n"
                "    python main.py\n\n"
                "Eso entrena y guarda los modelos necesarios."
            )

    # ------------------------------------------------------------------
    def _predict(self):
        # Leer y validar los valores ingresados
        values = {}
        out_of_range_msgs = []

        for internal_name, label_text, unit, _ in FIELDS:
            raw = self.entries[internal_name].get().strip()
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Valor inválido", f"'{label_text}' debe ser un número.")
                return

            if val < 0:
                messagebox.showerror("Valor inválido", f"'{label_text}' no puede ser negativo.")
                return

            values[internal_name] = val

            low, high = VALID_RANGES[internal_name]
            if val < low or val > high:
                out_of_range_msgs.append(f"{label_text} ({val}) fuera del rango típico [{low}–{high}]")

        # Elegir modelo
        use_xgb = self.model_choice.get().startswith("XGBoost")
        model = self.model_xgb if use_xgb else self.model_rf

        if model is None:
            messagebox.showerror(
                "Modelo no disponible",
                "El modelo seleccionado no está cargado. Corre 'python main.py' primero."
            )
            return

        X_new = pd.DataFrame([values])[[f[0] for f in FIELDS]]  # respeta el orden de columnas
        prediccion = model.predict(X_new)[0]

        self.result_label.config(text=f"Resistencia predicha: {prediccion:.2f} MPa")

        if out_of_range_msgs:
            self.warning_label.config(
                text="⚠ Advertencia — extrapolación fuera del rango de entrenamiento:\n"
                     + "; ".join(out_of_range_msgs)
            )
        else:
            self.warning_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConcreteStrengthApp(root)
    root.mainloop()
