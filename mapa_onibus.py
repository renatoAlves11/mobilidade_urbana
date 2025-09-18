# mapa_onibus_simplificado.py
# Versão reduzida: sem argparse, basta editar o caminho do .RData e executar.
# Requer: pip install pyreadr folium branca

import numpy as np
import pandas as pd
import pyreadr
import folium
from folium.plugins import HeatMap
from branca.element import Template, MacroElement

# ========= CONFIGURAÇÃO RÁPIDA =========
RDATA_PATH = "linha_455_ida_volta.RData"   # <- troque pelo seu arquivo .RData
OUT_HTML   = "mapa_linha_455.html"         # <- nome do HTML de saída

# Parâmetros de estabilidade (ajuste se necessário)
TERMINAL_RADIUS_M = 250.0   # raio (m) para considerar que chegou ao terminal
MIN_STEP_M        = 20.0    # distância mínima (m) para contar como evidência
MIN_SPEED         = None    # ex.: 1.0 (na unidade do VELOCITY) ou None p/ ignorar
K_BOOTSTRAP       = 3       # evidências seguidas p/ fixar 1º estado
K_FLIP            = 5       # evidências contrárias p/ trocar o estado

# ------------- Funções utilitárias -------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c  # km

def compute_terminals(df: pd.DataFrame):
    min_lat_idx = df['LATITUDE'].idxmin()
    max_lat_idx = df['LATITUDE'].idxmax()
    min_lon_idx = df['LONGITUDE'].idxmin()
    max_lon_idx = df['LONGITUDE'].idxmax()

    min_lat_point = df.loc[min_lat_idx, ['LONGITUDE', 'LATITUDE']].values
    max_lat_point = df.loc[max_lat_idx, ['LONGITUDE', 'LATITUDE']].values
    min_lon_point = df.loc[min_lon_idx, ['LONGITUDE', 'LATITUDE']].values
    max_lon_point = df.loc[max_lon_idx, ['LONGITUDE', 'LATITUDE']].values

    dist_lat = haversine(min_lat_point[1], min_lat_point[0], max_lat_point[1], max_lat_point[0])
    dist_lon = haversine(min_lon_point[1], min_lon_point[0], max_lon_point[1], max_lon_point[0])

    if dist_lat > dist_lon:
        A = min_lat_point
        B = max_lat_point
    else:
        A = min_lon_point
        B = max_lon_point
    return tuple(A), tuple(B)  # (lon, lat)

def classify_with_memory(
    df: pd.DataFrame, A, B,
    terminal_radius_m: float = TERMINAL_RADIUS_M,
    min_step_m: float = MIN_STEP_M,
    min_speed: float | None = MIN_SPEED,
    k_bootstrap: int = K_BOOTSTRAP,
    k_flip: int = K_FLIP
):
    A = np.array(A); B = np.array(B)
    vAB = B - A  # (lon, lat)

    n = len(df)
    evidence = np.zeros(n, dtype=int)
    for i in range(1, n-1):
        lon1, lat1 = df.iloc[i][['LONGITUDE','LATITUDE']]
        lon2, lat2 = df.iloc[i+1][['LONGITUDE','LATITUDE']]
        mv = np.array([lon2 - lon1, lat2 - lat1])
        step_m = haversine(lat1, lon1, lat2, lon2) * 1000.0

        if step_m < min_step_m:
            evidence[i] = 0
            continue
        if (min_speed is not None) and ('VELOCITY' in df.columns) and pd.notnull(df.iloc[i]['VELOCITY']):
            if df.iloc[i]['VELOCITY'] < min_speed:
                evidence[i] = 0
                continue

        dp = float(np.dot(mv, vAB))
        if dp > 0: evidence[i] = 1
        elif dp < 0: evidence[i] = -1
        else: evidence[i] = 0

    def near_terminal(lon, lat):
        dA = haversine(lat, lon, A[1], A[0]) * 1000.0
        dB = haversine(lat, lon, B[1], B[0]) * 1000.0
        return (dA <= terminal_radius_m) or (dB <= terminal_radius_m), (dA <= terminal_radius_m), (dB <= terminal_radius_m)

    state = 0   # 0 = indefinido, +1 = Ida, -1 = Volta
    opp_run = pos_run = neg_run = 0

    out = []
    for i in range(n):
        lon, lat = df.iloc[i][['LONGITUDE','LATITUDE']]
        at_term, atA, atB = near_terminal(lon, lat)

        if at_term:
            state = 1 if atA else (-1 if atB else 0)
            opp_run = pos_run = neg_run = 0

        e = evidence[i]

        if state == 0:
            if e == 1:
                pos_run += 1; neg_run = 0
                if pos_run >= k_bootstrap:
                    state = 1; pos_run = neg_run = 0
            elif e == -1:
                neg_run += 1; pos_run = 0
                if neg_run >= k_bootstrap:
                    state = -1; pos_run = neg_run = 0
        else:
            if (e == 0) or (e == state):
                opp_run = 0
            else:
                opp_run += 1
                if opp_run >= k_flip and not at_term:
                    state = -state
                    opp_run = 0

        out.append(state)

    label_map = {1: "Ida A → B", -1: "Volta B → A", 0: "Parado/Perpendicular"}
    return [label_map[s] for s in out]

def build_map(df: pd.DataFrame, A, B, out_html: str):
    center = [df["LATITUDE"].mean(), df["LONGITUDE"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    g_ida   = folium.FeatureGroup(name="Trajeto: Ida (A → B)", overlay=True, show=True)
    g_volta = folium.FeatureGroup(name="Trajeto: Volta (B → A)", overlay=True, show=True)
    g_term  = folium.FeatureGroup(name="Terminais", overlay=True, show=True)
    g_heat  = folium.FeatureGroup(name="Densidade de pontos", overlay=False, show=False)

    folium.Marker([A[1], A[0]], tooltip="Ponto A", icon=folium.Icon(color="black")).add_to(g_term)
    folium.Marker([B[1], B[0]], tooltip="Ponto B", icon=folium.Icon(color="orange")).add_to(g_term)

    color = {"Ida A → B": "green", "Volta B → A": "red"}
    arr = df[["LATITUDE", "LONGITUDE", "VECTOR_DIRECTION"]].values
    for i in range(len(arr) - 1):
        lat1, lon1, dir1 = arr[i]
        lat2, lon2, _    = arr[i + 1]
        seg = folium.PolyLine([[lat1, lon1], [lat2, lon2]], weight=3, opacity=0.9,
                              color=color.get(dir1, "blue"))
        if dir1 == "Ida A → B":
            seg.add_to(g_ida)
        elif dir1 == "Volta B → A":
            seg.add_to(g_volta)

    HeatMap(df[["LATITUDE", "LONGITUDE"]].values, radius=6, blur=8, min_opacity=0.3).add_to(g_heat)

    g_ida.add_to(m); g_volta.add_to(m); g_term.add_to(m); g_heat.add_to(m)

    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
       position: fixed;
       bottom: 30px; left: 30px; z-index: 9999;
       background: white; padding: 10px 12px; border:2px solid #444; border-radius: 6px;
       box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 14px;">
      <div style="font-weight:600; margin-bottom:6px;">Legenda</div>
      <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
        <span style="width:14px; height:4px; background:green; display:inline-block;"></span>
        <span>Ida (A → B)</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
        <span style="width:14px; height:4px; background:red; display:inline-block;"></span>
        <span>Volta (B → A)</span>
      </div>
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return out_html

# ------------- Execução simples -------------
res = pyreadr.read_r(RDATA_PATH)
df = (res['filtered_data'] if 'filtered_data' in res.keys() else res[list(res.keys())[0]]).copy()
df = df.sort_values('ID').reset_index(drop=True)

A, B = compute_terminals(df)
df['VECTOR_DIRECTION'] = classify_with_memory(df, A, B)

out_path = build_map(df, A, B, OUT_HTML)
print(f"✔ Mapa salvo em: {out_path}")
