# mapa_onibus_multilinhas_simplificado.py
# Gera UM ÚNICO mapa com TODAS as linhas informadas em FILES,
# com filtros por linha e por sentido (Ida/Volta) e terminais,
# e, adicionalmente, camadas com pontos MÉDIOS e MEDIANOS (Geral/Ida/Volta) por linha e no GERAL.
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

# Parâmetros
TERMINAL_RADIUS_M = 250.0
MIN_STEP_M        = 15.0

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

# --------- Classificação por RUN (soma de projeções i->i+1) ----------
def classify_by_route_runs(
    df: pd.DataFrame, A, B,
    terminal_radius_m: float = TERMINAL_RADIUS_M,
    min_step_m: float = MIN_STEP_M
):
    import numpy as np
    A = np.array(A, dtype=float)  # (lon, lat)
    B = np.array(B, dtype=float)
    vAB = B - A

    def dist_m_to(point_lon, point_lat, term):
        return haversine(point_lat, point_lon, term[1], term[0]) * 1000.0

    def which_terminal(lon, lat):
        dA = dist_m_to(lon, lat, A)
        dB = dist_m_to(lon, lat, B)
        if dA <= terminal_radius_m: return +1
        if dB <= terminal_radius_m: return -1
        return 0

    n = len(df)
    if n < 2:
        return ["Parado/Perpendicular"] * n

    seg_proj = np.zeros(n-1, dtype=float)
    for i in range(n-1):
        lon1, lat1 = float(df.iloc[i]["LONGITUDE"]), float(df.iloc[i]["LATITUDE"])
        lon2, lat2 = float(df.iloc[i+1]["LONGITUDE"]), float(df.iloc[i+1]["LATITUDE"])
        step_m = haversine(lat1, lon1, lat2, lon2) * 1000.0
        if step_m < min_step_m:
            seg_proj[i] = 0.0
            continue
        mv = np.array([lon2 - lon1, lat2 - lat1], dtype=float)
        seg_proj[i] = float(np.dot(mv, vAB))

    term_hits = []
    for i in range(n):
        lon, lat = float(df.iloc[i]["LONGITUDE"]), float(df.iloc[i]["LATITUDE"])
        t = which_terminal(lon, lat)
        if t != 0:
            term_hits.append(i)

    runs = []
    if len(term_hits) == 0:
        runs.append((0, n-1))
    else:
        first_hit = term_hits[0]
        if first_hit > 0:
            runs.append((0, first_hit))
        for a, b in zip(term_hits, term_hits[1:]):
            if b > a:
                runs.append((a, b))
        last_hit = term_hits[-1]
        if last_hit < n-1:
            runs.append((last_hit, n-1))

    labels = np.zeros(n, dtype=int)  # +1 ida, -1 volta, 0 indef
    for (s, e) in runs:
        total = float(np.nansum(seg_proj[s:e])) if e > s else 0.0
        if total > 0:
            lab = +1
        elif total < 0:
            lab = -1
        else:
            lon_s, lat_s = float(df.iloc[s]["LONGITUDE"]), float(df.iloc[s]["LATITUDE"])
            lon_e, lat_e = float(df.iloc[e]["LONGITUDE"]), float(df.iloc[e]["LATITUDE"])
            dAs = dist_m_to(lon_s, lat_s, A); dAe = dist_m_to(lon_e, lat_e, A)
            dBs = dist_m_to(lon_s, lat_s, B); dBe = dist_m_to(lon_e, lat_e, B)
            ida_evidence   = (dBe < dBs) and (dAe > dAs)
            volta_evidence = (dAe < dAs) and (dBe > dBs)
            if ida_evidence and not volta_evidence: lab = +1
            elif volta_evidence and not ida_evidence: lab = -1
            else: lab = +1 if dBe < dAe else -1
        labels[s:e+1] = lab

    map_label = {+1: "Ida A → B", -1: "Volta B → A", 0: "Parado/Perpendicular"}
    return [map_label[int(x)] for x in labels.tolist()]

# =================== NOVO: Centros (média/mediana) ===================
def centers_for_df(df: pd.DataFrame):
    """
    Retorna lista de:
      (escopo, mean_lat, mean_lon, median_lat, median_lon, n)
    com escopos: 'Geral', 'Ida', 'Volta'
    """
    out = []
    subsets = {
        "Geral": df,
        "Ida":   df[df["VECTOR_DIRECTION"] == "Ida A → B"],
        "Volta": df[df["VECTOR_DIRECTION"] == "Volta B → A"],
    }
    for escopo, sub in subsets.items():
        if len(sub) == 0:
            out.append((escopo, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        mean_lat = float(np.nanmean(sub["LATITUDE"]))
        mean_lon = float(np.nanmean(sub["LONGITUDE"]))
        med_lat  = float(np.nanmedian(sub["LATITUDE"]))
        med_lon  = float(np.nanmedian(sub["LONGITUDE"]))
        out.append((escopo, mean_lat, mean_lon, med_lat, med_lon, int(len(sub))))
    return out

# -------------------------- Mapa Folium ------------------------------
def build_map_multilines(datasets, out_html: str):
    all_lat = np.concatenate([d['df']['LATITUDE'].values for d in datasets if len(d['df'])>0])
    all_lon = np.concatenate([d['df']['LONGITUDE'].values for d in datasets if len(d['df'])>0])

    center = [float(np.nanmean(all_lat)), float(np.nanmean(all_lon))]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # coletores para centros gerais
    centers_general_rows = []

    for d in datasets:
        name = d['name']
        df = d['df']
        A  = d['A']
        B  = d['B']

        # camadas de trajetos
        g_ida   = folium.FeatureGroup(name=f"Linha {name} — Ida (A → B)", overlay=True, show=True)
        g_volta = folium.FeatureGroup(name=f"Linha {name} — Volta (B → A)", overlay=True, show=True)
        g_term  = folium.FeatureGroup(name=f"Terminais — {name}", overlay=True, show=True)

        # NOVO: camadas de centros por linha
        g_cent_mean   = folium.FeatureGroup(name=f"Centros (média) — {name}", overlay=True, show=False)
        g_cent_median = folium.FeatureGroup(name=f"Centros (mediana) — {name}", overlay=True, show=False)

        # terminais
        folium.Marker([A[1], A[0]], tooltip=f"Linha {name} — Ponto A", icon=folium.Icon(color="black")).add_to(g_term)
        folium.Marker([B[1], B[0]], tooltip=f"Linha {name} — Ponto B", icon=folium.Icon(color="orange")).add_to(g_term)

        # trajetos
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

        # centros (média/mediana) por linha
        centers = centers_for_df(df)
        # cores por escopo dos centros
        scope_color = {"Geral": "black", "Ida": "green", "Volta": "red"}

        for escopo, mean_lat, mean_lon, med_lat, med_lon, n in centers:
            if n == 0:  # nada a marcar
                continue
            # marcador de MÉDIA
            folium.Marker(
                location=[mean_lat, mean_lon],
                tooltip=f"{name} — {escopo} • MÉDIA",
                icon=folium.Icon(color=scope_color[escopo], icon="ok-sign"),
            ).add_to(g_cent_mean)
            # marcador de MEDIANA
            folium.Marker(
                location=[med_lat, med_lon],
                tooltip=f"{name} — {escopo} • MEDIANA",
                icon=folium.Icon(color=scope_color[escopo], icon="star"),
            ).add_to(g_cent_median)

            # acumula para centros gerais
            centers_general_rows.append({
                "escopo": escopo,
                "mean_lat": mean_lat, "mean_lon": mean_lon,
                "median_lat": med_lat, "median_lon": med_lon
            })

        # adiciona camadas da linha
        g_ida.add_to(m); g_volta.add_to(m); g_term.add_to(m)
        g_cent_mean.add_to(m); g_cent_median.add_to(m)

    # --------- Centros GERAIS (todas as linhas juntas) ----------
    if centers_general_rows:
        # agrega por escopo (média das médias e mediana das medianas por simplicidade)
        dfc = pd.DataFrame(centers_general_rows)
        # média geral dos centros (não dos pontos!): suficiente p/ referência visual
        mean_general = dfc.groupby("escopo", as_index=False)[["mean_lat","mean_lon"]].mean(numeric_only=True)
        median_general = dfc.groupby("escopo", as_index=False)[["median_lat","median_lon"]].median(numeric_only=True)

        g_cent_mean_all   = folium.FeatureGroup(name="Centros (média) — GERAL", overlay=True, show=False)
        g_cent_median_all = folium.FeatureGroup(name="Centros (mediana) — GERAL", overlay=True, show=False)

        scope_color = {"Geral": "black", "Ida": "green", "Volta": "red"}

        for _, row in mean_general.iterrows():
            escopo = row["escopo"]
            folium.Marker(
                location=[float(row["mean_lat"]), float(row["mean_lon"])],
                tooltip=f"GERAL — {escopo} • MÉDIA (dos centros)",
                icon=folium.Icon(color=scope_color[escopo], icon="ok-sign"),
            ).add_to(g_cent_mean_all)

        for _, row in median_general.iterrows():
            escopo = row["escopo"]
            folium.Marker(
                location=[float(row["median_lat"]), float(row["median_lon"])],
                tooltip=f"GERAL — {escopo} • MEDIANA (dos centros)",
                icon=folium.Icon(color=scope_color[escopo], icon="star"),
            ).add_to(g_cent_median_all)

        g_cent_mean_all.add_to(m)
        g_cent_median_all.add_to(m)

    # legenda fixa
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
       position: fixed;
       bottom: 30px; left: 30px; z-index: 9999;
       background: white; padding: 10px 12px; border:2px solid #444; border-radius: 6px;
       box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 14px;">
      <div style="font-weight:600; margin-bottom:6px;">Legenda</div>
      <div style="margin:4px 0;"><b>Cores</b>: <span style="color:green;">Ida</span>, <span style="color:red;">Volta</span>, <span>Preto</span> = Geral</div>
      <div style="margin:4px 0;"><b>Ícones</b>: ✔ = Média, ★ = Mediana</div>
      <div style="margin-top:6px;">Linhas: ative/desative camadas no painel (direita).</div>
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
