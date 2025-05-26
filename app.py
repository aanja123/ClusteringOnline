# app.py LOCAL: http://10.33.25.133:8050 ONLINE: https://clusteringonline.onrender.com/
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

# === Load All Cluster Data ===
datasets = {
    "Draga": {
        "pkl": "saved_clusters/audio_cluster_data_Random_Draga_hm.pkl",
        "folder": "data",
    },
    "Log": {
        "pkl": "saved_clusters/audio_cluster_data_Random_Log_hm05.pkl",
        "folder": "dataLog",
    },
    "Prekmurje": {
        "pkl": "saved_clusters/audio_cluster_data_Random_Prekmurje_hm05.pkl",
        "folder": "dataPrekmurje",
    }
}

for key, value in datasets.items():
    data = joblib.load(open(value["pkl"], "rb"))
    value["X_2d"] = data["X_2d_filtered"]
    value["labels"] = data["cluster_labels_filtered"]
    value["file_map"] = data["file_map_filtered"]
    value["cluster_preds"] = data["cluster_label_predictions"]

    # === New: Generate descriptive labels for legend ===
    cluster_labels_named = np.array([
        f"{value['cluster_preds'].get(lbl, [('Unknown', 0)])[0][0]} ({lbl})"
        if isinstance(value['cluster_preds'].get(lbl, [("Unknown", 0)])[0], tuple)
        else f"{value['cluster_preds'].get(lbl, ['Unknown'])[0]} ({lbl})"
        for lbl in value["labels"]
    ])

    # === Modified: use descriptive cluster labels in color ===
    value["fig"] = px.scatter(
        x=value["X_2d"][:, 0],
        y=value["X_2d"][:, 1],
        color=cluster_labels_named,  # <-- changed line
        labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Cluster"},
        title=key,
        hover_name=[f"{os.path.basename(f)} @ {s:.2f}s" for f, s in value["file_map"]],
    )

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Clustering"

app.layout = html.Div([
    html.H2("Clustering (click on points to play audio)"),
    html.Div([
        html.Button("Draga", id="btn-draga", n_clicks=0),
        html.Button("Log", id="btn-log", n_clicks=0),
        html.Button("Prekmurje", id="btn-prekmurje", n_clicks=0),
    ], style={"marginBottom": "20px"}),
    dcc.Store(id="current-dataset", data="Draga"),
    dcc.Graph(id="cluster-plot", figure=datasets["Draga"]["fig"]),
    html.Div(id="info-output"),
    html.Audio(id="audio-player", controls=True),
])

# === Callback to Switch Dataset ===
@app.callback(
    [Output("cluster-plot", "figure"),
     Output("current-dataset", "data")],
    [Input("btn-draga", "n_clicks"),
     Input("btn-log", "n_clicks"),
     Input("btn-prekmurje", "n_clicks")]
)
def update_plot(n_draga, n_log, n_prekmurje):
    ctx = dash.callback_context
    if not ctx.triggered:
        return datasets["Draga"]["fig"], "Draga"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "btn-log":
        return datasets["Log"]["fig"], "Log"
    elif button_id == "btn-prekmurje":
        return datasets["Prekmurje"]["fig"], "Prekmurje"
    return datasets["Draga"]["fig"], "Draga"

# === Callback for Point Click ===
@app.callback(
    [Output("info-output", "children"),
     Output("audio-player", "src")],
    [Input("cluster-plot", "clickData"),
     Input("current-dataset", "data")]
)
def on_point_click(clickData, dataset_name):
    if clickData is None:
        return "Click on a point to play audio.", None

    dataset = datasets[dataset_name]
    file_map = dataset["file_map"]
    labels = dataset["labels"]
    cluster_preds = dataset["cluster_preds"]
    folder = dataset["folder"]

    hover_text = clickData["points"][0].get("hovertext", "")
    try:
        hover_fname, hover_start_s = hover_text.split(" @ ")
        hover_start_s = float(hover_start_s.replace("s", ""))
    except:
        return "Error parsing hovertext.", None

    correct_index = None
    for i, (f, s) in enumerate(file_map):
        if os.path.basename(f) == hover_fname and abs(s - hover_start_s) < 1e-2:
            correct_index = i
            break
    if correct_index is None:
        return "Could not find matching point.", None

    file_name, start_time = file_map[correct_index]
    file_path = os.path.join(folder, os.path.basename(file_name))
    label = labels[correct_index]
    top_pred = cluster_preds.get(label, [("Unknown", 0)])
    top_class = top_pred[0][0] if isinstance(top_pred[0], tuple) else top_pred[0]

    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=2.0)
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(tmp_path, y, sr)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)
    audio_src = f"data:audio/wav;base64,{base64.b64encode(audio_bytes).decode('utf-8')}"

    info = html.Div([
        html.P(f"File: {file_path}"),
        html.P(f"Start Time: {start_time:.2f} s"),
        html.P(f"Cluster: {label}"),
        html.P(f"Top Prediction: {top_class}"),
    ])

    return info, audio_src

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)