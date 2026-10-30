"""Interfaz Gradio: consume la API por HTTP (no carga el modelo)."""
import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# (nombre, mínimo, máximo, valor inicial) — ejemplo de un grano SEKER
FIELDS = [
    ("Area", 1000, 500000, 28395), ("Perimeter", 100, 3000, 610.291),
    ("MajorAxisLength", 50, 1000, 208.178), ("MinorAxisLength", 50, 1000, 173.889),
    ("AspectRation", 1, 5, 1.197), ("Eccentricity", 0, 1, 0.55),
    ("ConvexArea", 1000, 500000, 28715), ("EquivDiameter", 30, 900, 190.141),
    ("Extent", 0, 1, 0.764), ("Solidity", 0, 1, 0.989),
    ("roundness", 0, 1, 0.958), ("Compactness", 0, 1, 0.913),
    ("ShapeFactor1", 0, 0.1, 0.00733), ("ShapeFactor2", 0, 0.01, 0.00314),
    ("ShapeFactor3", 0, 1, 0.834), ("ShapeFactor4", 0, 1, 0.999),
]


def clasificar(*values):
    payload = {f[0]: float(v) for f, v in zip(FIELDS, values)}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.RequestException:
        return "⚠️ La API no responde. Verifica que esté en ejecución.", {}
    if r.status_code != 200:
        return f"⚠️ Error {r.status_code}: {r.text[:200]}", {}
    out = r.json()
    return f"{out['variety']} (confianza {out['confidence']:.1%})", out["probabilities"]


with gr.Blocks(title="Dry Bean") as demo:
    gr.Markdown("# 🫘 Clasificador de frijol seco\nIngresa las 16 medidas del grano.")
    inputs = [gr.Number(label=n, value=v, minimum=lo, maximum=hi) for n, lo, hi, v in FIELDS]
    btn = gr.Button("Clasificar", variant="primary")
    label = gr.Textbox(label="Variedad predicha")
    probs = gr.Label(label="Probabilidades", num_top_classes=7)
    btn.click(clasificar, inputs, [label, probs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
