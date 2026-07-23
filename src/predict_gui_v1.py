"""
predict_gui_v1.py

Aplicación de escritorio (Tkinter) para predecir la resistencia a compresión
del concreto a partir de la dosificación de la mezcla, usando el modelo
XGBoost/Random Forest ya entrenado en T3 (main.py), y comparar el resultado
contra el método empírico ACI 211.1 (fórmula de Abrams calibrada en T3).

Incluye dos modos de entrada, intercambiables con un selector:
  - "Valores directos (kg/m³)": se ingresan las 8 variables tal como las
    espera el modelo.
  - "Proporción volumétrica (práctica peruana)": se ingresa la dosificación
    como se especifica típicamente en obra (bolsas de cemento por m³ y
    proporción de agregados "partes por bolsa", p. ej. 1:2:2), y el programa
    convierte esos valores a kg/m³ antes de predecir.

Requisito previo: haber corrido `python main.py` al menos una vez, para que
existan los modelos guardados en ../results/.

Uso:
    cd src
    python predict_gui_v1.py

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

# ---------------------------------------------------------------------
# Constantes del método empírico ACI 211.1 / Ley de Abrams
# (A, B calibrados en evaluate.py::evaluate_aci_baseline sobre el
#  conjunto de entrenamiento del proyecto — ver Informe Final, Sección II.E)
ACI_A = 69.11
ACI_B = 2.55

# Constantes para el modo de proporción volumétrica (práctica de obra Perú)
PESO_BOLSA_CEMENTO_KG = 42.5     # bolsa estándar de cemento en Perú
FT3_A_M3 = 0.0283168
DENSIDAD_AGREGADO_ASUMIDA = 1600.0  # kg/m3, densidad suelta aproximada (ajustable)

# Campos del modo "valores directos"
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


def proporcion_a_kg_m3(bolsas_m3, partes_gruesa, partes_fina, rel_ac):
    """
    Convierte una dosificación expresada en proporción volumétrica de obra
    (bolsas de cemento por m³ + partes de agregado por bolsa + relación a/c)
    a kg/m³, usando la convención de que 1 bolsa de cemento ocupa
    aproximadamente 1 pie³ suelto.

    NOTA: Es una aproximación con fines educativos/comparativos. No
    reemplaza un diseño de mezcla formal (ACI 211.1 completo con
    corrección por humedad y peso específico real de los agregados).
    """
    cemento = bolsas_m3 * PESO_BOLSA_CEMENTO_KG
    agua = cemento * rel_ac
    grueso = bolsas_m3 * partes_gruesa * FT3_A_M3 * DENSIDAD_AGREGADO_ASUMIDA
    fino = bolsas_m3 * partes_fina * FT3_A_M3 * DENSIDAD_AGREGADO_ASUMIDA
    return {"Cement": cemento, "Water": agua, "CoarseAggregate": grueso, "FineAggregate": fino}


def prediccion_aci(cemento_kg, agua_kg):
    """Estimación empírica ACI 211.1 / Ley de Abrams: f'c = A / B^(w/c)."""
    wc = agua_kg / cemento_kg
    return ACI_A / (ACI_B ** wc)


class ConcreteStrengthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Resistencia a Compresión del Concreto")
        self.root.geometry("520x760")
        self.root.resizable(False, False)

        self.entries = {}
        self.prop_entries = {}
        self.model_choice = tk.StringVar(value="XGBoost (recomendado)")
        self.input_mode = tk.StringVar(value="directo")

        self._build_ui()
        self._load_models()
        self._toggle_mode()

    # ------------------------------------------------------------------
    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Predicción de Resistencia a Compresión\ndel Concreto",
            font=("Segoe UI", 14, "bold"),
            pady=8,
        )
        title.pack()

        # ---- Selector de modo de entrada ----
        mode_frame = tk.LabelFrame(self.root, text="Modo de entrada", padx=10, pady=8)
        mode_frame.pack(fill="x", padx=20, pady=(4, 8))

        tk.Radiobutton(
            mode_frame, text="Valores directos (kg/m³)",
            variable=self.input_mode, value="directo", command=self._toggle_mode,
        ).pack(anchor="w")
        tk.Radiobutton(
            mode_frame, text="Proporción volumétrica (práctica de obra - Perú)",
            variable=self.input_mode, value="proporcion", command=self._toggle_mode,
        ).pack(anchor="w")

        # ---- Modo directo ----
        self.direct_frame = tk.Frame(self.root, padx=20)
        for internal_name, label_text, unit, default in FIELDS:
            row = tk.Frame(self.direct_frame, pady=3)
            row.pack(fill="x")
            tk.Label(row, text=f"{label_text}:", width=22, anchor="w", font=("Segoe UI", 9)).pack(side="left")
            entry = tk.Entry(row, width=10, justify="right")
            entry.insert(0, str(default))
            entry.pack(side="left", padx=5)
            tk.Label(row, text=unit, width=6, anchor="w", fg="gray40", font=("Segoe UI", 9)).pack(side="left")
            self.entries[internal_name] = entry

        # ---- Modo proporción ----
        self.prop_frame = tk.Frame(self.root, padx=20)

        prop_note = tk.Label(
            self.prop_frame,
            text="Ej.: 8.5 bolsas/m³, proporción 1:2:2 (grueso:fino), a/c=0.5\n"
                 "Se asume 1 bolsa de cemento ≈ 1 pie³ suelto, y densidad de\n"
                 "agregado suelto de 1600 kg/m³ (aproximación referencial).",
            font=("Segoe UI", 8, "italic"), fg="gray30", justify="left",
        )
        prop_note.pack(anchor="w", pady=(2, 8))

        prop_fields = [
            ("bolsas", "Bolsas de cemento por m³", "bolsas", 8.5),
            ("prop_gruesa", "Proporción Agregado Grueso", "partes/bolsa", 2.0),
            ("prop_fina", "Proporción Agregado Fino", "partes/bolsa", 2.0),
            ("rel_ac", "Relación Agua/Cemento (a/c)", "—", 0.5),
        ]
        for key, label_text, unit, default in prop_fields:
            row = tk.Frame(self.prop_frame, pady=3)
            row.pack(fill="x")
            tk.Label(row, text=f"{label_text}:", width=24, anchor="w", font=("Segoe UI", 9)).pack(side="left")
            entry = tk.Entry(row, width=10, justify="right")
            entry.insert(0, str(default))
            entry.pack(side="left", padx=5)
            tk.Label(row, text=unit, width=10, anchor="w", fg="gray40", font=("Segoe UI", 9)).pack(side="left")
            self.prop_entries[key] = entry

        # Campo de edad, compartido por ambos modos
        age_row = tk.Frame(self.prop_frame, pady=3)
        age_row.pack(fill="x")
        tk.Label(age_row, text="Edad de curado:", width=24, anchor="w", font=("Segoe UI", 9)).pack(side="left")
        self.prop_age_entry = tk.Entry(age_row, width=10, justify="right")
        self.prop_age_entry.insert(0, "28")
        self.prop_age_entry.pack(side="left", padx=5)
        tk.Label(age_row, text="días", width=10, anchor="w", fg="gray40", font=("Segoe UI", 9)).pack(side="left")

        self.prop_preview_label = tk.Label(
            self.prop_frame, text="", font=("Segoe UI", 8), fg="gray20", justify="left"
        )
        self.prop_preview_label.pack(anchor="w", pady=(8, 0))

        # ---- Selector de modelo ----
        model_frame = tk.Frame(self.root, pady=10)
        model_frame.pack(fill="x", padx=20)
        tk.Label(model_frame, text="Modelo:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Combobox(
            model_frame, textvariable=self.model_choice,
            values=["XGBoost (recomendado)", "Random Forest"],
            state="readonly", width=22,
        ).pack(side="left", padx=10)

        # ---- Botón ----
        tk.Button(
            self.root, text="Predecir Resistencia", command=self._predict,
            bg="#2E7D32", fg="white", font=("Segoe UI", 11, "bold"),
            padx=10, pady=8, cursor="hand2",
        ).pack(pady=14)

        # ---- Resultados ----
        results_frame = tk.LabelFrame(self.root, text="Resultados", padx=15, pady=10)
        results_frame.pack(fill="x", padx=20)

        self.ml_result_label = tk.Label(
            results_frame, text="Modelo ML: — MPa",
            font=("Segoe UI", 14, "bold"), fg="#1B5E20",
        )
        self.ml_result_label.pack(anchor="w")

        self.aci_result_label = tk.Label(
            results_frame, text="Método empírico ACI 211.1: — MPa",
            font=("Segoe UI", 11), fg="#5D4037",
        )
        self.aci_result_label.pack(anchor="w", pady=(4, 0))

        self.diff_label = tk.Label(
            results_frame, text="", font=("Segoe UI", 9, "italic"), fg="gray30",
        )
        self.diff_label.pack(anchor="w", pady=(4, 0))

        self.warning_label = tk.Label(
            self.root, text="", font=("Segoe UI", 8), fg="#B71C1C",
            wraplength=460, justify="center",
        )
        self.warning_label.pack(pady=(8, 0))

        tk.Label(
            self.root,
            text="Modelo ML entrenado sobre el dataset Concrete Compressive Strength (UCI).\n"
                 "Herramienta de apoyo académico — no reemplaza el ensayo normativo NTP 339.034 / ASTM C39\n"
                 "ni un diseño de mezcla formal ACI 211.1.",
            font=("Segoe UI", 7), fg="gray50", justify="center",
        ).pack(side="bottom", pady=8)

    # ------------------------------------------------------------------
    def _toggle_mode(self):
        if self.input_mode.get() == "directo":
            self.prop_frame.pack_forget()
            self.direct_frame.pack(fill="x")
        else:
            self.direct_frame.pack_forget()
            self.prop_frame.pack(fill="x")

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
    def _read_float(self, entry, label_text):
        raw = entry.get().strip()
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"'{label_text}' debe ser un número.")

    # ------------------------------------------------------------------
    def _get_values_direct(self):
        values = {}
        for internal_name, label_text, _, _ in FIELDS:
            val = self._read_float(self.entries[internal_name], label_text)
            if val < 0:
                raise ValueError(f"'{label_text}' no puede ser negativo.")
            values[internal_name] = val
        return values

    def _get_values_from_proportion(self):
        bolsas = self._read_float(self.prop_entries["bolsas"], "Bolsas de cemento por m³")
        prop_gruesa = self._read_float(self.prop_entries["prop_gruesa"], "Proporción Agregado Grueso")
        prop_fina = self._read_float(self.prop_entries["prop_fina"], "Proporción Agregado Fino")
        rel_ac = self._read_float(self.prop_entries["rel_ac"], "Relación Agua/Cemento")
        edad = self._read_float(self.prop_age_entry, "Edad de curado")

        if min(bolsas, prop_gruesa, prop_fina, rel_ac, edad) < 0:
            raise ValueError("Ningún valor puede ser negativo.")

        derivados = proporcion_a_kg_m3(bolsas, prop_gruesa, prop_fina, rel_ac)

        self.prop_preview_label.config(
            text=f"→ Equivalente en kg/m³: Cemento={derivados['Cement']:.1f}, "
                 f"Agua={derivados['Water']:.1f}, Grueso={derivados['CoarseAggregate']:.1f}, "
                 f"Fino={derivados['FineAggregate']:.1f}"
        )

        values = {
            "Cement": derivados["Cement"],
            "BlastFurnaceSlag": 0.0,
            "FlyAsh": 0.0,
            "Water": derivados["Water"],
            "Superplasticizer": 0.0,
            "CoarseAggregate": derivados["CoarseAggregate"],
            "FineAggregate": derivados["FineAggregate"],
            "Age": edad,
        }
        return values

    # ------------------------------------------------------------------
    def _predict(self):
        try:
            if self.input_mode.get() == "directo":
                values = self._get_values_direct()
            else:
                values = self._get_values_from_proportion()
        except ValueError as e:
            messagebox.showerror("Valor inválido", str(e))
            return

        # Advertencias de rango (siempre se evalúan sobre los kg/m3 resultantes)
        out_of_range_msgs = []
        for internal_name, (low, high) in VALID_RANGES.items():
            val = values[internal_name]
            if val < low or val > high:
                out_of_range_msgs.append(f"{internal_name} ({val:.1f}) fuera de [{low}-{high}]")

        use_xgb = self.model_choice.get().startswith("XGBoost")
        model = self.model_xgb if use_xgb else self.model_rf
        if model is None:
            messagebox.showerror("Modelo no disponible", "Corre 'python main.py' primero para generar los modelos.")
            return

        X_new = pd.DataFrame([values])[[f[0] for f in FIELDS]]
        pred_ml = model.predict(X_new)[0]
        pred_aci = prediccion_aci(values["Cement"], values["Water"])

        self.ml_result_label.config(text=f"Modelo ML ({self.model_choice.get().split(' ')[0]}): {pred_ml:.2f} MPa")
        self.aci_result_label.config(text=f"Método empírico ACI 211.1 (Abrams): {pred_aci:.2f} MPa")

        diff = pred_ml - pred_aci
        signo = "por encima" if diff >= 0 else "por debajo"
        self.diff_label.config(
            text=f"El modelo ML predice {abs(diff):.2f} MPa {signo} de la estimación empírica tradicional."
        )

        if out_of_range_msgs:
            self.warning_label.config(
                text="⚠ Advertencia — valores fuera del rango de entrenamiento (posible extrapolación):\n"
                     + "; ".join(out_of_range_msgs)
            )
        else:
            self.warning_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConcreteStrengthApp(root)
    root.mainloop()
