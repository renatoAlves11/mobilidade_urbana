# mapa_onibus_multilinhas_simplificado.py
# Gera UM ÚNICO mapa com TODAS as linhas informadas em FILES,
# com filtros por linha e por sentido (Ida/Volta) e terminais.
# Classificação robusta: variação das distâncias aos terminais (dA/dB) com suavização.
# Requer: pip install pyreadr folium branca

import os
import numpy as np
import pandas as pd
import pyreadr
import folium
from branca.element import Template, MacroElement

# ========= CONFIGURAÇÃO RÁPIDA =========
FILES = [
    "linha_232_ida_volta.RData",
    "linha_390_ida_volta.RData",
    "linha_455_ida_volta.RData",
]
OUT_HTML = "mapa_todas_linhas.html"

# Parâmetros da classificação (ajuste se necessário)
SMOOTH_WINDOW     = 5      # pontos para mediana móvel
DELTA_THR_M       = 10.0   # variação mínima (m) em dA/dB p/ considerar evidência
TERMINAL_RADIUS_M = 250.0  # raio (m) para detectar terminal
MIN_STEP_M        = 15.0   # passo mínimo (m) p/ considerar movimento

# ----------------------------- Haversine -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c  # km

# ---------------------- Terminais pelos extremos ---------------------
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

# --------------- Classificação por distância aos terminais ----------
def classify_by_route_runs(
    df: pd.DataFrame, A, B,
    terminal_radius_m: float = TERMINAL_RADIUS_M,
    min_step_m: float = MIN_STEP_M
):
    """
    Rotula por TRECHOS (runs) entre terminais.
    Para cada run (do terminal X até o próximo terminal Y), soma as projeções dos segmentos no eixo A→B.
    - Soma > 0  => run inteiro 'Ida A → B'
    - Soma < 0  => run inteiro 'Volta B → A'
    Se o dataset não começa/termina exatamente em terminal, rotula o trecho inicial/final pela soma também.
    """
    import numpy as np

    # vetor eixo A->B em coordenadas (lon, lat)
    A = np.array(A, dtype=float)  # (lon, lat)
    B = np.array(B, dtype=float)
    vAB = B - A

    # helper: distância terminal
    def dist_m_to(point_lon, point_lat, term):
        return haversine(point_lat, point_lon, term[1], term[0]) * 1000.0

    def which_terminal(lon, lat):
        dA = dist_m_to(lon, lat, A)
        dB = dist_m_to(lon, lat, B)
        if dA <= terminal_radius_m:
            return +1  # terminal A
        if dB <= terminal_radius_m:
            return -1  # terminal B
        return 0      # nenhum

    n = len(df)
    if n < 2:
        return ["Parado/Perpendicular"] * n

    # 1) compute projeção ao longo de A->B para cada segmento i->i+1 (com filtro de passo)
    seg_proj = np.zeros(n-1, dtype=float)  # projeção assinada
    for i in range(n-1):
        lon1, lat1 = float(df.iloc[i]["LONGITUDE"]), float(df.iloc[i]["LATITUDE"])
        lon2, lat2 = float(df.iloc[i+1]["LONGITUDE"]), float(df.iloc[i+1]["LATITUDE"])
        step_m = haversine(lat1, lon1, lat2, lon2) * 1000.0
        if step_m < min_step_m:
            seg_proj[i] = 0.0
            continue
        mv = np.array([lon2 - lon1, lat2 - lat1], dtype=float)
        seg_proj[i] = float(np.dot(mv, vAB))  # projeção (escala arbitrária; o sinal é o que importa)

    # 2) encontre "marcos" de terminal (índices), para segmentar runs
    term_hits = []
    for i in range(n):
        lon, lat = float(df.iloc[i]["LONGITUDE"]), float(df.iloc[i]["LATITUDE"])
        t = which_terminal(lon, lat)
        if t != 0:
            term_hits.append(i)

    # 3) constroi os runs: pares [start_idx, end_idx] (inclusive), cada um termina em um terminal
    runs = []
    if len(term_hits) == 0:
        # não passou por terminal: um único run do começo ao fim
        runs.append((0, n-1))
    else:
        # pode haver um trecho antes do primeiro terminal detectado
        first_hit = term_hits[0]
        if first_hit > 0:
            runs.append((0, first_hit))

        # runs completos entre hits consecutivos
        for a, b in zip(term_hits, term_hits[1:]):
            if b > a:
                runs.append((a, b))

        # pode haver um trecho depois do último terminal
        last_hit = term_hits[-1]
        if last_hit < n-1:
            runs.append((last_hit, n-1))

    # 4) para cada run, soma seg_proj dentro do run e rotula todo o run pelo sinal da soma
    labels = np.zeros(n, dtype=int)  # +1 ida, -1 volta, 0 indef
    for (s, e) in runs:
        # segmentos válidos do run: i = s .. e-1
        if e > s:
            total = float(np.nansum(seg_proj[s:e]))
        else:
            total = 0.0

        if total > 0:
            lab = +1  # Ida A → B
        elif total < 0:
            lab = -1  # Volta B → A
        else:
            # empate: decide pelo terminal de origem/destino se houver, senão pelo delta de dA/dB
            lon_s, lat_s = float(df.iloc[s]["LONGITUDE"]), float(df.iloc[s]["LATITUDE"])
            lon_e, lat_e = float(df.iloc[e]["LONGITUDE"]), float(df.iloc[e]["LATITUDE"])
            dAs = dist_m_to(lon_s, lat_s, A); dAe = dist_m_to(lon_e, lat_e, A)
            dBs = dist_m_to(lon_s, lat_s, B); dBe = dist_m_to(lon_e, lat_e, B)
            # se aproximou de B e afastou de A → ida; vice-versa → volta
            ida_evidence   = (dBe < dBs) and (dAe > dAs)
            volta_evidence = (dAe < dAs) and (dBe > dBs)
            if ida_evidence and not volta_evidence:
                lab = +1
            elif volta_evidence and not ida_evidence:
                lab = -1
            else:
                # fallback: compara distâncias finais (mais perto de qual terminal?)
                lab = +1 if dBe < dAe else -1

        labels[s:e+1] = lab  # o run inteiro recebe a mesma etiqueta

    # 5) mapeia para strings finais
    map_label = {+1: "Ida A → B", -1: "Volta B → A", 0: "Parado/Perpendicular"}
    return [map_label[int(x)] for x in labels.tolist()]

# -------------------------- Mapa Folium ------------------------------
def build_map_multilines(datasets, out_html: str):
    all_lat = np.concatenate([d['df']['LATITUDE'].values for d in datasets if len(d['df'])>0])
    all_lon = np.concatenate([d['df']['LONGITUDE'].values for d in datasets if len(d['df'])>0])

    center = [float(np.nanmean(all_lat)), float(np.nanmean(all_lon))]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    for d in datasets:
        name = d['name']
        df = d['df']
        A  = d['A']
        B  = d['B']

        g_ida   = folium.FeatureGroup(name=f"Linha {name} — Ida (A → B)", overlay=True, show=True)
        g_volta = folium.FeatureGroup(name=f"Linha {name} — Volta (B → A)", overlay=True, show=True)
        g_term  = folium.FeatureGroup(name=f"Terminais — {name}", overlay=True, show=True)

        folium.Marker([A[1], A[0]], tooltip=f"Linha {name} — Ponto A", icon=folium.Icon(color="black")).add_to(g_term)
        folium.Marker([B[1], B[0]], tooltip=f"Linha {name} — Ponto B", icon=folium.Icon(color="orange")).add_to(g_term)

        color = {"Ida A → B": "green", "Volta B → A": "red"}
        arr = df[["LATITUDE","LONGITUDE","VECTOR_DIRECTION"]].values

        for i in range(len(arr)-1):
            lat1, lon1, dir1 = arr[i]
            lat2, lon2, _    = arr[i+1]
            seg = folium.PolyLine([[lat1, lon1],[lat2, lon2]], weight=3, opacity=0.85,
                                  color=color.get(dir1, "blue"))
            if dir1 == "Ida A → B":
                seg.add_to(g_ida)
            elif dir1 == "Volta B → A":
                seg.add_to(g_volta)

        g_ida.add_to(m); g_volta.add_to(m); g_term.add_to(m)

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

# ------------------------------ Execução -----------------------------
datasets = []
for path in FILES:
    if not os.path.exists(path):
        print(f"[AVISO] Arquivo não encontrado: {path}")
        continue
    try:
        res = pyreadr.read_r(path)
        df = (res['filtered_data'] if 'filtered_data' in res.keys() else res[list(res.keys())[0]]).copy()
        df = df.sort_values('ID').reset_index(drop=True)
        A, B = compute_terminals(df)
        df['VECTOR_DIRECTION'] = classify_by_route_runs(df, A, B)
        name = os.path.splitext(os.path.basename(path))[0]
        datasets.append({'name': name, 'df': df, 'A': A, 'B': B})
        ida_ct = (df['VECTOR_DIRECTION'] == 'Ida A → B').sum()
        vol_ct = (df['VECTOR_DIRECTION'] == 'Volta B → A').sum()
        print(f"[OK] {name}  ->  Ida: {ida_ct}  Volta: {vol_ct}  Total: {len(df)}")
    except Exception as e:
        print(f"[ERRO] {path}: {e}")

if not datasets:
    raise SystemExit("Nenhum dataset válido foi carregado. Verifique a lista FILES.")

out_path = build_map_multilines(datasets, OUT_HTML)
print(f"✔ Mapa salvo em: {out_path}")
