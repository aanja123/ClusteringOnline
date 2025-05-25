# app.py OPEN: http://10.33.25.133:8050 
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import joblib
import librosa
import soundfile as sf
import tempfile
import base64
import numpy as np
import os

# === Load Cluster Data ===
data_path = "/home/anjaalocal/storage/ClusteringOnline/saved_clusters/audio_cluster_data_Random_Draga_hm.pkl"
data = joblib.load(open(data_path, "rb"))

X_2d = data["X_2d_filtered"]
labels = data["cluster_labels_filtered"]
file_map = data["file_map_filtered"]
cluster_preds = data["cluster_label_predictions"]

# === Create Scatter Plot ===
fig = px.scatter(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    color=labels.astype(str),
    labels={"x": "UMAP 1", "y": "UMAP 2"},
    title="Draga (hm)",
    hover_name=[f"{os.path.basename(f)} @ {s:.2f}s" for f, s in file_map],
)

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Clustering"

app.layout = html.Div([
    html.H2("Clustering"),
    dcc.Graph(id="cluster-plot", figure=fig),
    html.Div(id="info-output"),
    html.Audio(id="audio-player", controls=True),
])

# === Callback for Point Click ===
@app.callback(
    [Output("info-output", "children"),
     Output("audio-player", "src")],
    [Input("cluster-plot", "clickData")]
)
def on_point_click(clickData):
    if clickData is None:
        return "Click on a point to play audio.", None

    point_idx = clickData["points"][0]["pointIndex"]
    file_name, start_time = file_map[point_idx]
    label = labels[point_idx]
    top_pred = cluster_preds.get(label, [("Unknown", 0)])
    top_class = top_pred[0][0] if isinstance(top_pred[0], tuple) else top_pred[0]

    # Load audio segment
    y, sr = librosa.load(file_name, sr=None, offset=start_time, duration=2.0)
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(tmp_path, y, sr)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_src = f"data:audio/wav;base64,{b64_audio}"

    info = html.Div([
        html.P(f"File: {file_name}"),
        html.P(f"Start Time: {start_time:.2f} s"),
        html.P(f"Cluster: {label}"),
        html.P(f"Top Prediction: {top_class}"),
    ])

    return info, audio_src

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
