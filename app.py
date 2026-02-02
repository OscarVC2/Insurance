import os
import re
import json
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher


import streamlit as st

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Insurance Calculator"
MODEL = "gpt-5.2"  # Ajusta si tu cuenta usa otro alias/nombre
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash" )
DEFAULT_CHATGPT_WEIGHT = float(os.getenv("CHATGPT_WEIGHT", "0.75"))
DATA_DIR = Path(__file__).parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CLAIMS_PATH = DATA_DIR / "claims.json"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Brand / Models (dropdown)
# Default marca: "Indica la marca"
# -----------------------------
BRAND_MODELS = {
    "Indica la marca": [],
    "Toyota": [
        "Hilux",
        "Land Cruiser Pick Up (Serie 70)",
        "Stout 2026",
        "Land Cruiser Prado",
        "RAV4 Hybrid",
        "Corolla Cross",
        "Fortuner",
        "Rush",
        "Corolla",
        "Yaris",
        "Agya",
        "Hiace",
        "Coaster",
    ],
    "Hyundai": [
        "Creta",
        "Tucson",
        "Santa Fe",
        "Kona",
        "Venue",
        "Santa Cruz",
        "Elantra",
        "Accent",
        "Ioniq 5",
        "Ioniq 6",
    ],
    "Kia": [
        "Sportage",
        "Sportage Híbrida 2026",
        "Sorento",
        "Seltos",
        "Sonet",
        "Carens",
        "K3",
        "K3 Cross",
        "Picanto",
        "Soluto",
        "K2700",
        "K3000S",
    ],
    "Nissan": [
        "Frontier S",
        "Frontier XE",
        "Frontier PRO-4X",
        "Kicks",
        "Qashqai",
        "X-Trail e-Power",
        "Pathfinder",
        "Patrol",
        "Versa",
        "Sentra",
    ],
    "Ford": [
        "F-150",
        "F-150 Raptor",
        "Ranger XL",
        "Ranger XLT",
        "Ranger Wildtrak",
        "Ranger Raptor",
        "Maverick",
        "Explorer",
        "Expedition",
        "Territory",
        "Bronco",
        "Bronco Sport",
        "Everest",
    ],
    "Mitsubishi": [
        "L200 (Nueva generación)",
        "Montero Sport",
        "Outlander",
        "Outlander Sport",
        "Eclipse Cross",
        "Xpander Cross",
    ],
    "Honda": [
        "CR-V",
        "HR-V",
        "Civic",
        "Odyssey",
    ],
    "Isuzu": [
        "D-Max",
        "Serie N (camión)",
    ],
    "BMW": [
        "Serie 3",
        "X1",
        "X5",
    ],
    "Mercedes-Benz": [
        "Clase C",
        "GLE",
    ],
    "Audi": [
        "Q3",
        "Q5",
    ],
    "MG": [
        "ZS",
        "RX5",
    ],
    "Changan": [
        "CS35",
        "CS55",
    ],
    "GWM": [
        "Poer",
        "Haval Jolion",
    ],
    "BYD": [
        "Dolphin",
        "Yuan Plus",
    ],
    "Otro": [],
}

# -----------------------------
# UI / CSS
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

CSS = """
<style>
:root { --card: rgba(255,255,255,0.06); --border: rgba(255,255,255,0.08); --text: rgba(255,255,255,0.92); --muted: rgba(255,255,255,0.65); }
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 30% 0%, rgba(255,0,0,0.10), transparent 50%),
              radial-gradient(1000px 600px at 80% 10%, rgba(0,120,255,0.10), transparent 50%),
              linear-gradient(180deg, #0b0f14, #070a0f 60%, #070a0f);
  color: var(--text);
}
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b0f14, #070a0f); border-right: 1px solid var(--border); }
h1, h2, h3 { color: var(--text); letter-spacing: 0.2px; }
.small-muted { color: var(--muted); font-size: 0.9rem; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}
.kpi {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 18px;
}
.badge {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.05);
  color: rgba(255,255,255,0.85);
}
.badge-blue { background: rgba(59,130,246,0.15); border-color: rgba(59,130,246,0.25); }
.badge-yellow { background: rgba(234,179,8,0.12); border-color: rgba(234,179,8,0.22); }
.badge-green { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.22); }
.badge-red { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.22); }

.stButton button {
  border-radius: 14px !important;
  padding: 0.75rem 1rem !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
.stButton button[kind="primary"] {
  background: linear-gradient(135deg, #ff4d4d, #ff2a2a) !important;
  border: 0 !important;
}
hr { border-color: rgba(255,255,255,0.08); }
.block-title { font-size: 1.1rem; font-weight: 700; margin: 0 0 10px 0; }
pre, code { white-space: pre-wrap !important; word-break: break-word !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers: persistence
# -----------------------------
def load_claims():
    if CLAIMS_PATH.exists():
        try:
            return json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_claims(claims):
    CLAIMS_PATH.write_text(
        json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def next_claim_id(claims):
    # CLM-0001 ...
    n = 1
    existing = {c.get("id") for c in claims}
    while True:
        cid = f"CLM-{n:04d}"
        if cid not in existing:
            return cid
        n += 1


# -----------------------------
# Helpers: Streamlit image compatibility
# -----------------------------
def st_image_safe(img, caption=None):
    """
    Compatibilidad:
    - Streamlit nuevo: use_container_width
    - Streamlit viejo: use_column_width
    """
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


# -----------------------------
# Output sanitation:
# - Quita líneas "##" sueltas
# - Arregla tablas markdown "pegadas" en una sola línea para que rendericen como tabla
# -----------------------------
def strip_lonely_hash_lines(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if re.fullmatch(r"\s*#{1,6}\s*", line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def ensure_table_newline_after_heading(text: str) -> str:
    # Si el modelo pega "TABLA ... HNL | Ítem | ..." en la misma línea, fuerza salto de línea antes del header
    return re.sub(
        r"(?i)(TABLA DE PRESUPUESTO DETALLADO EN HNL)\s*(\|)",
        r"\1\n\n\2",
        text or "",
    )


def rebuild_runon_budget_table(block: str) -> str:
    """
    Repara tablas markdown "run-on" donde el modelo devuelve todo en una sola línea.
    Reconstruye:
      - Encabezado "4. TABLA DE PRESUPUESTO..."
      - Header de tabla
      - Separador
      - Filas (7 columnas)
    """
    if not block:
        return block

    s = re.sub(r"\s+", " ", block).strip()

    heading = ""
    m_heading = re.search(
        r"(?i)(\b[1-8]\.\s*TABLA DE PRESUPUESTO DETALLADO EN HNL\b)", s
    )
    if m_heading:
        heading = m_heading.group(1)
        s = s.replace(heading, "", 1).strip()

    m = re.search(
        r"(\|\s*Ítem\s*\|.*?\|\s*Subtotal\s*\(HNL\)\s*\|)", s, flags=re.I
    )
    if not m:
        return block

    header = m.group(1).strip()
    rest = s[m.end() :].strip()

    # Tokeniza celdas por '|', elimina celdas vacías y las de alineación (---, :---:, etc)
    raw_cells = [c.strip() for c in rest.split("|")]
    cells = []
    for c in raw_cells:
        if not c:
            continue
        if re.fullmatch(r"-{3,}:?", c):
            continue
        if re.fullmatch(r":?-{3,}:?", c):
            continue
        cells.append(c)

    # Encuentra el inicio de la primera fila (ítem numérico y luego texto)
    start = None
    for i in range(len(cells) - 6):
        if cells[i].isdigit() and not cells[i + 1].isdigit():
            start = i
            break
    if start is None:
        return block

    def clean_int(x: str) -> str:
        d = re.sub(r"\D", "", x or "")
        return d if d else "0"

    sep = "|---:|---|---|---:|---:|---:|---:|"
    out = []
    if heading:
        out.append(heading)
        out.append("")

    out.append(header)
    out.append(sep)

    for j in range(start, len(cells), 7):
        row = cells[j : j + 7]
        if len(row) < 7:
            break
        if not row[0].isdigit():
            break

        row[3] = clean_int(row[3])
        row[4] = clean_int(row[4])
        row[5] = clean_int(row[5])
        row[6] = clean_int(row[6])

        out.append("| " + " | ".join(row) + " |")

    return "\n".join(out).strip()


def normalize_budget_tables(text: str) -> str:
    """
    Detecta bloques con header '| Ítem | ... |' y los repara si vienen "pegados".
    """
    if not text:
        return text

    lines = text.splitlines()
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if "| Ítem" in line:
            block_lines = [line]
            j = i + 1
            while j < len(lines) and ("|" in lines[j]) and lines[j].strip() != "":
                block_lines.append(lines[j])
                j += 1

            # Heurística de "mal formado": header + separador pegados, o demasiados pipes en una línea
            malformed = False
            for bl in block_lines:
                if ("| Ítem" in bl and "|---" in bl):
                    malformed = True
                if bl.count("|") > 40:
                    malformed = True
                if len(re.findall(r"\|\s*\d+\s*\|", bl)) > 1:
                    malformed = True

            block = "\n".join(block_lines)
            if malformed:
                rebuilt = rebuild_runon_budget_table(block)
                out.extend(rebuilt.splitlines())
            else:
                out.extend(block_lines)

            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def sanitize_model_output(text: str) -> str:
    text = strip_lonely_hash_lines(text or "")
    text = ensure_table_newline_after_heading(text)
    text = normalize_budget_tables(text)
    return text.strip()


def escape_numbered_headings_for_markdown(text: str) -> str:
    """
    Evita que Markdown trate "4. TABLA ..." como lista numerada (lo cual puede romper tablas).
    Se muestra igual visualmente: "4. ..."
    """
    return re.sub(r"(?m)^(\s*)([1-8])\.\s+", r"\1\2\. ", text or "")


# -----------------------------
# OpenAI client / API key resolution
# -----------------------------
def get_api_key():
    # 1) secrets.toml (si existe)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        # no revienta: sigue
        pass

    # 2) ENV
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 3) input manual (sidebar)
    return None

def get_gemini_key():
    # 1) secrets.toml (si existe)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        pass

    # 2) ENV
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    return None


def get_openai_client(api_key: str):
    # Import aquí para que el app cargue aunque no esté instalado
    from openai import OpenAI

    return OpenAI(api_key=api_key)


# -----------------------------
# Prompt: reglas duras (para evitar totales = 0)
# -----------------------------
HARD_RULES = """
Eres un perito automotriz virtual especializado en estimación de daños por siniestros viales en Honduras.

REGLAS DURAS (OBLIGATORIAS):
1) Debes devolver EXACTAMENTE 8 secciones con estos encabezados EXACTOS (en mayúsculas):
   1. RESUMEN DEL SINIESTRO
   2. DAÑOS IDENTIFICADOS POR ZONA
   3. DECISIÓN TÉCNICA POR PIEZA
   4. TABLA DE PRESUPUESTO DETALLADO EN HNL
   5. TOTAL ESTIMADO DE REPARACIÓN
   6. CLASIFICACIÓN DEL DAÑO
   7. NIVEL DE CONFIANZA
   8. ADVERTENCIAS TÉCNICAS

2) En la sección 4, la tabla debe ser Markdown con EXACTAMENTE estas columnas:
   | Ítem | Pieza/Trabajo | Acción (Reparar/Reemplazar) | Costo pieza (HNL) | Mano de obra (HNL) | Pintura (HNL) | Subtotal (HNL) |

3) La tabla DEBE incluir saltos de línea reales:
   - Encabezado en una línea
   - Separador en una línea
   - Cada fila en su propia línea

4) TODOS los valores numéricos deben ir como enteros SIN comas, SIN signo de moneda, SIN texto extra.
   Ejemplo: 15850 (NO "15,850" y NO "L 15,850" y NO "15,850 HNL").

5) En la sección 5, debes escribir en una sola línea EXACTA:
   TOTAL ESTIMADO DE REPARACIÓN: L <NUMERO_ENTERO>
   Ejemplo: TOTAL ESTIMADO DE REPARACIÓN: L 15850

6) No uses "?" ni rangos en la línea del total. Si deseas escenarios, colócalos como texto adicional debajo, pero SIEMPRE conserva la línea exacta del total.

7) Prohibido terminar el output con líneas sueltas de Markdown como "##" o "#".
"""

def build_prompt(vehicle, accident, enfoque: str):
    base_info = f"""Datos del vehículo:
- Marca: {vehicle.get("marca") or "No especificado"}
- Modelo: {vehicle.get("modelo") or "No especificado"}
- Año: {vehicle.get("anio") or "No especificado"}
- Tipo: {vehicle.get("tipo") or "No especificado"}
- Combustible: {vehicle.get("combustible") or "No especificado"}
- Transmisión: {vehicle.get("transmision") or "No especificado"}

Datos del siniestro (opcional):
- Tipo de impacto: {accident.get("tipo_impacto") or "No especificado"}
- ¿Enciende?: {accident.get("enciende") or "No especificado"}
- Airbags: {accident.get("airbags") or "No especificado"}
"""
    if enfoque == "conservador":
        enfoque_txt = """ENFOQUE CONSERVADOR:
- Prioriza REPARAR sobre REEMPLAZAR si es técnicamente viable.
- Usa repuestos genéricos/alternativos comunes.
- Asume mejor escenario dentro de lo visible (no inventes daños ocultos extensos).
"""
    else:
        enfoque_txt = """ENFOQUE DETALLADO:
- Considera REEMPLAZAR cuando la reparación comprometa durabilidad/seguridad.
- Incluye daños ocultos PROBABLES según el tipo de impacto (sin exagerar).
- Mantén costos realistas del mercado hondureño.
"""
    return f"""{HARD_RULES}

Analiza las imágenes del vehículo siniestrado.

{base_info}

{enfoque_txt}

Devuelve el informe siguiendo exactamente las reglas duras.
"""


# -----------------------------
# Parsing / totals
# -----------------------------
SECTION_HEADERS = [
    ("resumen", r"1\.\s*RESUMEN DEL SINIESTRO"),
    ("danos", r"2\.\s*DAÑOS IDENTIFICADOS POR ZONA"),
    ("decisiones", r"3\.\s*DECISIÓN TÉCNICA POR PIEZA"),
    ("presupuesto", r"4\.\s*TABLA DE PRESUPUESTO DETALLADO EN HNL"),
    ("total", r"5\.\s*TOTAL ESTIMADO DE REPARACIÓN"),
    ("clasificacion", r"6\.\s*CLASIFICACIÓN DEL DAÑO"),
    ("confianza", r"7\.\s*NIVEL DE CONFIANZA"),
    ("advertencias", r"8\.\s*ADVERTENCIAS TÉCNICAS"),
]


def split_sections(text: str):
    out = {k: "" for k, _ in SECTION_HEADERS}
    if not text:
        return out

    matches = []
    for key, pat in SECTION_HEADERS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            matches.append((key, m.start()))

    if not matches:
        return out

    matches.sort(key=lambda x: x[1])
    for i, (key, start) in enumerate(matches):
        end = matches[i + 1][1] if i + 1 < len(matches) else len(text)
        out[key] = text[start:end].strip()

    return out
def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sim(a: str, b: str) -> float:
    a2, b2 = _norm(a), _norm(b)
    if not a2 or not b2:
        return 0.0
    seq = SequenceMatcher(None, a2, b2).ratio()
    ta, tb = set(a2.split()), set(b2.split())
    j = (len(ta & tb) / max(1, len(ta | tb)))
    return max(seq, j)

def parse_budget_rows(report_text: str):
    """
    Extrae filas de la tabla de presupuesto del reporte (sección 4).
    Retorna lista de dicts con: pieza, accion, costo_pieza, mano, pintura, subtotal.
    """
    sec = split_sections(report_text).get("presupuesto", "") or ""
    lines = sec.splitlines()

    # buscar header de tabla
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and "item" in _norm(ln) and "subtotal" in _norm(ln):
            start = i
            break
    if start is None:
        return []

    rows = []
    for ln in lines[start+2:]:
        if not ln.strip().startswith("|"):
            break
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cols) != 7:
            continue
        if not cols[0].isdigit():
            continue
        try:
            pieza = cols[1]
            accion = cols[2]
            costo_pieza = int(re.sub(r"\D", "", cols[3]) or "0")
            mano = int(re.sub(r"\D", "", cols[4]) or "0")
            pintura = int(re.sub(r"\D", "", cols[5]) or "0")
            # subtotal del modelo se ignora: lo recalculamos
            subtotal = costo_pieza + mano + pintura
            rows.append({
                "pieza": pieza,
                "accion": accion,
                "costo_pieza": costo_pieza,
                "mano": mano,
                "pintura": pintura,
                "subtotal": subtotal,
            })
        except Exception:
            continue

    return rows

def build_budget_table_md(rows):
    header = "| Ítem | Pieza/Trabajo | Acción (Reparar/Reemplazar) | Costo pieza (HNL) | Mano de obra (HNL) | Pintura (HNL) | Subtotal (HNL) |"
    sep = "|---:|---|---|---:|---:|---:|---:|"
    out = [header, sep]
    for i, r in enumerate(rows, start=1):
        subtotal = int(r["costo_pieza"]) + int(r["mano"]) + int(r["pintura"])
        out.append(
            f"| {i} | {r['pieza']} | {r['accion']} | {int(r['costo_pieza'])} | {int(r['mano'])} | {int(r['pintura'])} | {subtotal} |"
        )
    return "\n".join(out)

def blend_budget_rows(openai_rows, gemini_rows, w_chatgpt: float):
    """
    Baseline: filas OpenAI.
    Si Gemini tiene fila similar, mezcla costos por fila: w*openai + (1-w)*gemini.
    """
    if not openai_rows:
        return [], 0

    used = set()
    blended = []

    for r in openai_rows:
        best_j = None
        best_s = 0.0

        for j, g in enumerate(gemini_rows or []):
            if j in used:
                continue
            s = _sim(r["pieza"], g["pieza"])
            if s > best_s:
                best_s = s
                best_j = j

        if best_j is not None and best_s >= 0.72:
            g = gemini_rows[best_j]
            used.add(best_j)

            def mix(a, b):  # int
                return int(round(w_chatgpt * a + (1.0 - w_chatgpt) * b))

            costo_pieza = mix(r["costo_pieza"], g["costo_pieza"])
            mano = mix(r["mano"], g["mano"])
            pintura = mix(r["pintura"], g["pintura"])

            blended.append({
                "pieza": r["pieza"],
                "accion": r["accion"],
                "costo_pieza": costo_pieza,
                "mano": mano,
                "pintura": pintura,
            })
        else:
            # Sin match Gemini: conserva OpenAI tal cual (más peso a ChatGPT = baseline)
            blended.append({
                "pieza": r["pieza"],
                "accion": r["accion"],
                "costo_pieza": r["costo_pieza"],
                "mano": r["mano"],
                "pintura": r["pintura"],
            })

    total = sum(int(x["costo_pieza"]) + int(x["mano"]) + int(x["pintura"]) for x in blended)
    return blended, total

def replace_section_4_and_5(openai_report: str, table_md: str, total_int: int) -> str:
    # Reemplaza sección 4 completa
    out = re.sub(
        r"(?is)4\.\s*TABLA DE PRESUPUESTO DETALLADO EN HNL.*?(?=\n\s*5\.\s*TOTAL ESTIMADO DE REPARACIÓN)",
        "4. TABLA DE PRESUPUESTO DETALLADO EN HNL\n\n" + table_md.strip() + "\n",
        openai_report,
    )

    # Reemplaza sección 5 completa
    out = re.sub(
        r"(?is)5\.\s*TOTAL ESTIMADO DE REPARACIÓN.*?(?=\n\s*6\.\s*CLASIFICACIÓN DEL DAÑO)",
        "5. TOTAL ESTIMADO DE REPARACIÓN\nTOTAL ESTIMADO DE REPARACIÓN: L " + str(int(total_int)) + "\n",
        out,
    )
    return out

def merge_extra_bullets(openai_report: str, gemini_report: str, section_key: str, title: str) -> str:
    """
    Agrega bullets únicos de Gemini dentro de la sección indicada, sin tocar el resto.
    """
    if not gemini_report:
        return openai_report

    oa_sec = split_sections(openai_report).get(section_key, "") or ""
    gm_sec = split_sections(gemini_report).get(section_key, "") or ""
    if not oa_sec or not gm_sec:
        return openai_report

    oa_lines = set(_norm(x) for x in oa_sec.splitlines() if x.strip().startswith(("-", "•")))
    extras = []
    for ln in gm_sec.splitlines():
        if ln.strip().startswith(("-", "•")):
            if _norm(ln) not in oa_lines:
                extras.append(ln.strip())

    if not extras:
        return openai_report

    patch = oa_sec.rstrip() + "\n\n" + title + "\n" + "\n".join(f"- {x.lstrip('-• ').strip()}" for x in extras)

    # Sustituye SOLO esa sección completa en el reporte
    key_pat = dict(SECTION_HEADERS)[section_key]
    # rango: desde header de la sección hasta el inicio de la siguiente sección
    idx = [k for k, _ in SECTION_HEADERS].index(section_key)
    next_key = [k for k, _ in SECTION_HEADERS][idx + 1] if idx + 1 < len(SECTION_HEADERS) else None
    next_pat = dict(SECTION_HEADERS).get(next_key) if next_key else None

    if next_pat:
        out = re.sub(
            rf"(?is)({key_pat}.*?)(?=\n\s*{next_pat})",
            patch,
            openai_report,
        )
        return out

    return openai_report

def consensus_report(openai_report: str, gemini_report: str, w_chatgpt: float) -> str:
    if not gemini_report:
        return openai_report

    oa_rows = parse_budget_rows(openai_report)
    gm_rows = parse_budget_rows(gemini_report)

    blended_rows, blended_total = blend_budget_rows(oa_rows, gm_rows, w_chatgpt)
    table_md = build_budget_table_md(blended_rows)

    out = replace_section_4_and_5(openai_report, table_md, blended_total)

    # opcional: enriquecer daños/advertencias con bullets únicos Gemini (sin reescribir todo)
    out = merge_extra_bullets(out, gemini_report, "danos", "Notas adicionales (Gemini):")
    out = merge_extra_bullets(out, gemini_report, "advertencias", "Notas adicionales (Gemini):")

    return sanitize_model_output(out)

def extract_total_int(text: str) -> int:
    """
    Prioridad:
    1) Línea exacta "TOTAL ESTIMADO DE REPARACIÓN: L 12345"
    2) Suma de columna Subtotal de la tabla (si se puede)
    """
    if not text:
        return 0

    m = re.search(
        r"TOTAL ESTIMADO DE REPARACIÓN:\s*L\s*([0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    total = 0
    for line in text.splitlines():
        if line.strip().startswith("|") and line.count("|") >= 7:
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) == 7:
                if cols[0].isdigit() and cols[-1].isdigit():
                    total += int(cols[-1])
    return total


def compute_average_report(cons_text: str, det_text: str):
    cons_sections = split_sections(cons_text)
    det_sections = split_sections(det_text)

    cons_total = extract_total_int(cons_text)
    det_total = extract_total_int(det_text)
    avg_total = int(round((cons_total + det_total) / 2)) if (cons_total or det_total) else 0
    diff = abs(cons_total - det_total) if (cons_total and det_total) else 0
    pct = (diff / avg_total * 100) if avg_total else 0.0

    promedio = {
        "resumen": (
            "📊 ANÁLISIS COMPARATIVO DUAL\n\n"
            f"🔹 Estimación Conservadora: L {cons_total}\n"
            f"🔹 Estimación Detallada:   L {det_total}\n\n"
            f"📈 Diferencia: L {diff}\n"
            f"📊 Variación: {pct:.1f}%\n"
        ),
        "danos": (
            "CONSENSO DE DAÑOS IDENTIFICADOS:\n\n"
            + (cons_sections.get("danos", "") or "")
            + "\n\n--- Perspectiva adicional (Detallado) ---\n\n"
            + (det_sections.get("danos", "") or "")
        ),
        "decisiones": (
            "RECOMENDACIONES TÉCNICAS BALANCEADAS:\n\n"
            + (det_sections.get("decisiones", "") or cons_sections.get("decisiones", ""))
        ),
        "presupuesto": (
            "TABLA (Conservador):\n\n"
            + (cons_sections.get("presupuesto", "") or "")
            + "\n\nTABLA (Detallado):\n\n"
            + (det_sections.get("presupuesto", "") or "")
        ),
        "total": (
            f"💰 TOTAL ESTIMADO PROMEDIO: L {avg_total}\n\n"
            f"Rango sugerido: L {min(cons_total, det_total)} - L {max(cons_total, det_total)}"
        ),
        "clasificacion": (det_sections.get("clasificacion", "") or cons_sections.get("clasificacion", "")),
        "confianza": (
            "NIVEL DE CONFIANZA: MEDIO-ALTO (análisis dual)\n\n"
            f"Consistencia: {pct:.1f}% de variación."
        ),
        "advertencias": (det_sections.get("advertencias", "") or cons_sections.get("advertencias", "")),
        "_debug": {"cons_total": cons_total, "det_total": det_total, "avg_total": avg_total},
    }
    return cons_sections, det_sections, promedio


# -----------------------------
# OpenAI call (Responses API) with images
# -----------------------------
def file_to_data_url(uploaded_file) -> str:
    b = uploaded_file.read()
    uploaded_file.seek(0)
    mime = uploaded_file.type or "image/jpeg"
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def run_estimate(client, prompt: str, uploaded_files):
    """
    Sends prompt + images to OpenAI Responses API.
    """
    content = [{"type": "input_text", "text": prompt}]
    for uf in uploaded_files:
        content.append({"type": "input_image", "image_url": file_to_data_url(uf)})

    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
    )

    raw = (resp.output_text or "").strip()
    # FIX: sanitiza para:
    # - eliminar "##" sueltos
    # - normalizar tablas pegadas para que rendericen bien y puedan parsearse
    return sanitize_model_output(raw)

def run_estimate_gemini(prompt: str, uploaded_files, api_key: str) -> str:
    """
    Ejecuta Gemini multimodal (prompt + imágenes) y devuelve texto sanitizado.
    Requiere: pip install -U google-generativeai pillow
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("Falta google-generativeai. Instala: pip install -U google-generativeai pillow") from e

    genai.configure(api_key=api_key)

    try:
        from PIL import Image
        has_pil = True
    except Exception:
        has_pil = False

    model = genai.GenerativeModel(GEMINI_MODEL)

    parts = [prompt]
    for uf in uploaded_files:
        b = uf.getvalue()
        if has_pil:
            try:
                img = Image.open(BytesIO(b)).convert("RGB")
                parts.append(img)
            except Exception:
                # si una imagen falla, continúa sin romper
                continue

    resp = model.generate_content(parts)
    text = getattr(resp, "text", "") or ""
    return sanitize_model_output(text)


# -----------------------------
# Session init
# -----------------------------
if "claims" not in st.session_state:
    st.session_state.claims = load_claims()

if "nav" not in st.session_state:
    st.session_state.nav = "Reportar / Estimador"

if "selected_claim_id" not in st.session_state:
    st.session_state.selected_claim_id = None

if "analysis" not in st.session_state:
    st.session_state.analysis = {
        "cons_text": "",
        "det_text": "",
        "cons": None,
        "det": None,
        "avg": None,
        "ready": False,
    }

# -----------------------------
# Sidebar navigation (3 sections)
# -----------------------------
with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.markdown('<div class="small-muted">Navegación</div>', unsafe_allow_html=True)
    nav = st.radio(
        "Sección",
        ["Reportar / Estimador", "Mis Siniestros", "Aprobaciones"],
        index=["Reportar / Estimador", "Mis Siniestros", "Aprobaciones"].index(st.session_state.nav),
        label_visibility="collapsed",
    )
    st.session_state.nav = nav

    st.markdown("---")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            help="Opcional si no usas secrets.toml o variable de entorno",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------
# KPI helpers from claims
# -----------------------------
def approvals_kpis(claims):
    total = len(claims)
    approved = sum(1 for c in claims if c.get("approval_status") == "Aprobado")
    pending = total - approved
    return total, approved, pending


def claim_status_badge(c):
    s = c.get("approval_status", "Pendiente")
    lvl = c.get("approval_level", 2)
    if s == "Aprobado":
        return f"<span class='badge badge-green'>Aprobado · Nivel {lvl}</span>"
    return f"<span class='badge badge-yellow'>Pendiente · Nivel {lvl}</span>"


# -----------------------------
# Pages
# -----------------------------
def page_estimador():
    st.markdown("## Reportar / Estimador")
    st.markdown(
        '<div class="small-muted">Formulario centrado. Análisis dual: Conservador + Detallado + Promedio.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    left, mid, right = st.columns([1, 6, 1])
    with mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### Fotografías del vehículo")
        uploaded = st.file_uploader(
            "Sube fotos (frontal, trasera, laterales, close-ups)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("#### Previsualización")
            cols = st.columns(4)
            for i, f in enumerate(uploaded[:12]):
                with cols[i % 4]:
                    st_image_safe(f, caption=f.name)

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("### Datos del vehículo")
        c1, c2, c3 = st.columns(3)

        brands = list(BRAND_MODELS.keys())
        with c1:
            marca = st.selectbox("Marca", brands, index=0)
        with c2:
            modelos = ["Indica el modelo"] + (BRAND_MODELS.get(marca, []) if marca != "Indica la marca" else [])
            modelo = st.selectbox("Modelo", modelos, index=0)
        with c3:
            anio = st.text_input("Año", placeholder="Ej: 2018")

        c4, c5, c6 = st.columns(3)
        with c4:
            tipo = st.selectbox("Tipo", ["", "Sedán", "SUV", "Pick-up", "Hatchback", "Van"])
        with c5:
            combustible = st.selectbox("Combustible", ["", "Gasolina", "Diésel", "Híbrido", "Eléctrico"])
        with c6:
            transmision = st.selectbox("Transmisión", ["", "Manual", "Automática"])

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Datos del siniestro (opcional)")
        a1, a2, a3 = st.columns(3)
        with a1:
            tipo_impacto = st.selectbox(
                "Tipo de impacto",
                ["", "Frontal", "Trasero", "Lateral izquierdo", "Lateral derecho", "Volcadura", "Múltiple"],
            )
        with a2:
            enciende = st.selectbox("¿El vehículo enciende?", ["", "Sí enciende", "No enciende"])
        with a3:
            airbags = st.selectbox("Estado de airbags", ["", "No activados", "Activados"])

        st.markdown("<hr/>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns([2, 2, 2])

        can_analyze = bool(uploaded) and bool(get_api_key() or os.getenv("OPENAI_API_KEY"))
        with b1:
            run_btn = st.button("Analizar (Conservador + Detallado)", type="primary", disabled=not can_analyze)
        with b2:
            clear_btn = st.button("Limpiar resultados")
        with b3:
            can_save = bool(st.session_state.analysis.get("ready"))
            save_btn = st.button("Guardar como siniestro", disabled=not can_save)

        if clear_btn:
            st.session_state.analysis = {"cons_text": "", "det_text": "", "cons": None, "det": None, "avg": None, "ready": False}
            st.success("Resultados limpiados.")
            st.rerun()

        if run_btn:
            api_key_local = get_api_key() or os.getenv("OPENAI_API_KEY")
            if not api_key_local:
                st.error("Falta OPENAI_API_KEY (secrets.toml, ENV o input en sidebar).")
                return

            gemini_key_local = get_gemini_key() or os.getenv("GEMINI_API_KEY")
            w_chatgpt = float(st.session_state.get("chatgpt_weight", DEFAULT_CHATGPT_WEIGHT))

            client = get_openai_client(api_key_local)

            vehicle = {
                "marca": None if marca == "Indica la marca" else marca,
                "modelo": None if modelo == "Indica el modelo" else modelo,
                "anio": anio.strip(),
                "tipo": tipo,
                "combustible": combustible,
                "transmision": transmision,
            }
            accident = {"tipo_impacto": tipo_impacto, "enciende": enciende, "airbags": airbags}

            # --- CONSERVADOR ---
            with st.spinner("Generando estimación CONSERVADORA (ChatGPT + Gemini)..."):
                prompt_cons = build_prompt(vehicle, accident, "conservador")

                openai_cons = run_estimate(client, prompt_cons, uploaded)
                gemini_cons = ""
                if gemini_key_local:
                    try:
                        gemini_cons = run_estimate_gemini(prompt_cons, uploaded, gemini_key_local)
                    except Exception as e:
                        # si Gemini falla, no rompemos: seguimos con OpenAI
                        st.warning(f"Gemini falló, se usará solo ChatGPT. Detalle: {e}")

                cons_text = consensus_report(openai_cons, gemini_cons, w_chatgpt)

            # --- DETALLADO ---
            with st.spinner("Generando estimación DETALLADA (ChatGPT + Gemini)..."):
                prompt_det = build_prompt(vehicle, accident, "detallado")

                openai_det = run_estimate(client, prompt_det, uploaded)
                gemini_det = ""
                if gemini_key_local:
                    try:
                        gemini_det = run_estimate_gemini(prompt_det, uploaded, gemini_key_local)
                    except Exception as e:
                        st.warning(f"Gemini falló, se usará solo ChatGPT. Detalle: {e}")

                det_text = consensus_report(openai_det, gemini_det, w_chatgpt)

            # Promedio entre CONSENSOS (no entre proveedores)
            cons, det, avg = compute_average_report(cons_text, det_text)

            cons_total = extract_total_int(cons_text)
            det_total = extract_total_int(det_text)
            ready = (cons_total > 0 or det_total > 0) and (
                    ("TABLA DE PRESUPUESTO" in cons_text.upper()) or ("TABLA DE PRESUPUESTO" in det_text.upper())
            )

            st.session_state.analysis = {
                "cons_text": cons_text,
                "det_text": det_text,
                "cons": cons,
                "det": det,
                "avg": avg,
                "ready": ready,
                # opcional: guarda fuentes por auditoría
                "sources": {
                    "conservador": {"openai": openai_cons, "gemini": gemini_cons},
                    "detallado": {"openai": openai_det, "gemini": gemini_det},
                    "weights": {"chatgpt": w_chatgpt, "gemini": 1.0 - w_chatgpt},
                },
            }

            if not ready:
                st.warning("No se detectó TOTAL/tabla válida. Reintenta (formato estricto activo).")

            st.rerun()

        analysis = st.session_state.analysis
        if analysis.get("cons") or analysis.get("det") or analysis.get("avg"):
            st.markdown("<hr/>", unsafe_allow_html=True)

            t1, t2, t3 = st.tabs(["Promedio", "Conservador", "Detallado"])

            def render_section(title, content, render_as_markdown=False):
                if not content:
                    return
                with st.expander(title, expanded=True):
                    if render_as_markdown:
                        # FIX: renderiza Markdown real para que la tabla se vea como tabla
                        md = escape_numbered_headings_for_markdown(content)
                        st.markdown(md)
                    else:
                        st.code(content, language="text")

            with t1:
                st.markdown("### Informe Promedio")
                avg = analysis.get("avg")
                if avg:
                    render_section("Resumen del siniestro", avg.get("resumen", ""))
                    render_section("Daños identificados por zona", avg.get("danos", ""))
                    render_section("Decisiones técnicas por pieza", avg.get("decisiones", ""))
                    # FIX: presupuesto como markdown (tablas renderizables)
                    render_section("Presupuesto detallado", avg.get("presupuesto", ""), render_as_markdown=True)
                    render_section("Total estimado", avg.get("total", ""))
                    render_section("Clasificación del daño", avg.get("clasificacion", ""))
                    render_section("Nivel de confianza", avg.get("confianza", ""))
                    render_section("Advertencias técnicas", avg.get("advertencias", ""))

                    dbg = avg.get("_debug", {})
                    st.markdown(
                        f"<div class='small-muted'>Debug totales: cons={dbg.get('cons_total',0)} · det={dbg.get('det_total',0)} · avg={dbg.get('avg_total',0)}</div>",
                        unsafe_allow_html=True,
                    )

            with t2:
                st.markdown("### Informe Conservador (raw)")
                render_section("Salida completa", analysis.get("cons_text", ""), render_as_markdown=True)

            with t3:
                st.markdown("### Informe Detallado (raw)")
                render_section("Salida completa", analysis.get("det_text", ""), render_as_markdown=True)

        st.markdown("</div>", unsafe_allow_html=True)


def page_mis_siniestros():
    claims = st.session_state.claims
    total, approved, pending = approvals_kpis(claims)

    st.markdown("## Mis Siniestros")
    st.markdown(
        '<div class="small-muted">Este listado se alimenta 100% de “Guardar como siniestro”. Si no hay siniestros, aparece vacío.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"<div class='kpi'><div style='font-size:28px;font-weight:800'>{total}</div><div class='small-muted'>Total</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='kpi'><div style='font-size:28px;font-weight:800'>{approved}</div><div class='small-muted'>Aprobados</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='kpi'><div style='font-size:28px;font-weight:800'>{pending}</div><div class='small-muted'>Pendientes</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### Listado")
    st.markdown("")

    if not claims:
        st.info("No hay siniestros guardados.")
        return

    for c in reversed(claims):
        title = "Vehículo"
        veh = c.get("vehicle", {})
        acc = c.get("accident", {})
        vtxt = " ".join([x for x in [veh.get("marca"), veh.get("modelo"), veh.get("anio")] if x]).strip() or "Sin datos"
        ctxt = acc.get("tipo_impacto") or "Sin tipo de impacto"
        bid = c.get("id")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        top = st.columns([6, 2])
        with top[0]:
            st.markdown(f"### {title}")
            st.markdown(f"<div class='small-muted'>{vtxt} · {ctxt}</div>", unsafe_allow_html=True)
            st.markdown(claim_status_badge(c), unsafe_allow_html=True)
        with top[1]:
            st.markdown(
                f"<div style='text-align:right; font-weight:700; color: rgba(255,255,255,0.7);'>{bid}</div>",
                unsafe_allow_html=True,
            )

        b1, b2 = st.columns([2, 2])
        with b1:
            if st.button("Ver detalle", key=f"detail_{bid}"):
                st.session_state.selected_claim_id = bid
                st.rerun()
        with b2:
            if st.button("Ir Aprobaciones", key=f"goto_appr_{bid}"):
                st.session_state.nav = "Aprobaciones"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

    if st.session_state.selected_claim_id:
        cid = st.session_state.selected_claim_id
        found = next((x for x in claims if x.get("id") == cid), None)
        if found:
            st.markdown("---")
            st.markdown("## Detalle del Siniestro")
            st.markdown(f"<span class='badge badge-blue'>ID: {cid}</span>", unsafe_allow_html=True)
            st.markdown("")

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Daños")
            st.markdown(claim_status_badge(found), unsafe_allow_html=True)

            veh = found.get("vehicle", {})
            acc = found.get("accident", {})
            st.markdown("#### Descripción")
            st.write(
                {
                    "vehículo": veh,
                    "siniestro": acc,
                    "totales": found.get("analysis", {}).get("totals", {}),
                }
            )

            st.markdown("#### Historial de Aprobaciones")
            for h in found.get("approval_history", []):
                st.write(h)

            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Cerrar detalle"):
                st.session_state.selected_claim_id = None
                st.rerun()


def page_aprobaciones():
    claims = st.session_state.claims

    st.markdown("## Aprobaciones")
    st.markdown(
        '<div class="small-muted">Se alimenta 100% desde Mis Siniestros. Si no hay siniestros, todo queda en 0.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    total, approved, pending = approvals_kpis(claims)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"<div class='kpi'><div style='font-size:28px;font-weight:800'>{total}</div><div class='small-muted'>Total</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='kpi'><div style='font-size:28px;font-weight:800'>{approved}</div><div class='small-muted'>Aprobadas</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='kpi'><div style='font-size:28px;font-weight:800'>{pending}</div><div class='small-muted'>Pendientes</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### Pendientes")
    st.markdown("")

    if not claims:
        st.info("No hay aprobaciones porque no hay siniestros.")
        return

    pendings = [c for c in claims if c.get("approval_status") != "Aprobado"]
    if not pendings:
        st.success("No hay pendientes.")
        return

    for c in pendings:
        bid = c.get("id")
        veh = c.get("vehicle", {})
        acc = c.get("accident", {})
        vtxt = " ".join([x for x in [veh.get("marca"), veh.get("modelo"), veh.get("anio")] if x]).strip() or "Sin datos"
        ctxt = acc.get("tipo_impacto") or "Sin tipo de impacto"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {vtxt}")
        st.markdown(f"<div class='small-muted'>ID: {bid} · {ctxt}</div>", unsafe_allow_html=True)
        st.markdown(claim_status_badge(c), unsafe_allow_html=True)

        b1, b2 = st.columns([2, 2])
        with b1:
            if st.button("Abrir detalle", key=f"appr_detail_{bid}"):
                st.session_state.selected_claim_id = bid
                st.session_state.nav = "Mis Siniestros"
                st.rerun()

        with b2:
            if st.button("Marcar Aprobado", key=f"approve_{bid}"):
                for x in st.session_state.claims:
                    if x.get("id") == bid:
                        x["approval_status"] = "Aprobado"
                        x["approval_level"] = 2
                        hist = x.get("approval_history", [])
                        for h in hist:
                            if h.get("nivel") == 2 and h.get("estado") == "Pendiente":
                                h["estado"] = "Aprobado"
                                h["por"] = "Aprobador Nivel 2"
                                h["fecha"] = datetime.now().isoformat(timespec="seconds")
                        x["approval_history"] = hist
                        break

                save_claims(st.session_state.claims)
                st.success(f"{bid} aprobado.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")


# -----------------------------
# Router
# -----------------------------
if st.session_state.nav == "Reportar / Estimador":
    page_estimador()
elif st.session_state.nav == "Mis Siniestros":
    page_mis_siniestros()
else:
    page_aprobaciones()
