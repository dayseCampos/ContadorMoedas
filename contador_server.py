import os, secrets, io, base64, socket
from datetime import datetime
from flask import Flask, request, abort, send_file, render_template_string, redirect, url_for
import qrcode
import numpy as np
import cv2
from sklearn.cluster import KMeans

# ===================== UTIL: IP local =====================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ===================== PIPELINE COINSUM =====================
BASE_REF_WIDTH = 1107
BASE_MIN_R, BASE_MAX_R = 36, 140
BASE_MIN_DIST = 65
BASE_P2 = 32
P1 = 110

def grayworld_wb(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    k = (mb + mg + mr) / 3.0
    b *= k/(mb+1e-6); g *= k/(mg+1e-6); r *= k/(mr+1e-6)
    return np.clip(cv2.merge([b,g,r]), 0, 255).astype(np.uint8)

def clahe_bgr(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def ring_mask(shape, cx, cy, r, inner=0.70, outer=0.98):
    H, W = shape[:2]; Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return (d >= r*inner) & (d <= r*outer)

def circle_edge_score(img_bgr, x, y, r):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = ring_mask(gray.shape, x, y, r, 0.70, 0.98)
    vals = mag[m]
    return float(vals.mean()) if vals.size else 0.0

def detect_coins(img_bgr):
    h, w = img_bgr.shape[:2]
    k = w / float(BASE_REF_WIDTH)
    minR = max(10, int(round(BASE_MIN_R * k)))
    maxR = int(round(BASE_MAX_R * k))
    minDist = int(round(BASE_MIN_DIST * k))
    p2 = max(24, int(round(BASE_P2 * k)))

    img_pp = clahe_bgr(grayworld_wb(img_bgr))
    gray = cv2.cvtColor(img_pp, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.1, minDist=minDist,
        param1=P1, param2=p2, minRadius=minR, maxRadius=maxR
    )

    dets = []
    if circles is not None:
        for (x, y, r) in np.uint16(np.around(circles[0])):
            x, y, r = int(x), int(y), int(r)
            score = circle_edge_score(img_pp, x, y, r)
            dets.append((x, y, r, score))
    if not dets:
        return []

    K = min(6, len(dets))
    XY = np.array([[x, y] for (x, y, r, s) in dets], dtype=np.float32)
    labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(XY)

    best = {}
    for (x, y, r, s), lbl in zip(dets, labels):
        cur = best.get(lbl)
        if cur is None or s > cur[3]:
            best[lbl] = (x, y, r, s)
    picked = list(best.values())
    picked.sort(key=lambda t: -t[3])
    return [(x, y, r) for (x, y, r, _) in picked]

def coin_features(img_bgr, x, y, r):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    A, B = lab[:,:,1], lab[:,:,2]
    h, w = img_bgr.shape[:2]; Y, X = np.ogrid[:h, :w]
    d2 = (X-x)**2 + (Y-y)**2
    m_coin  = d2 <= r*r
    m_inner = d2 <= int(0.58*r)**2
    m_ring  = (d2 <= int(0.98*r)**2) & (d2 >= int(0.70*r)**2)
    def mmean(M, Mmask):
        vals = M[Mmask]; return float(vals.mean()) if vals.size else 0.0
    h_coin, s_coin, v_coin = mmean(H,m_coin), mmean(S,m_coin), mmean(V,m_coin)
    h_in, s_in, v_in = mmean(H,m_inner), mmean(S,m_inner), mmean(V,m_inner)
    h_rg, s_rg, v_rg = mmean(H,m_ring),  mmean(S,m_ring),  mmean(V,m_ring)
    a_coin, b_coin = mmean(A,m_coin), mmean(B,m_coin)
    ring_diff = np.linalg.norm([h_in-h_rg, s_in-s_rg, v_in-v_rg])
    return {"x":x,"y":y,"r":r,"h":h_coin,"s":s_coin,"v":v_coin,
            "h_in":h_in,"s_in":s_in,"v_in":v_in,
            "h_rg":h_rg,"s_rg":s_rg,"v_rg":v_rg,
            "a":a_coin,"b":b_coin,"ring_diff":ring_diff}

def classify_all(feats):
    out = []
    def is_one(c):
        neut_inner = abs(c["a"] - 128) + abs(c["b"] - 128) < 22
        ring_more_sat = (c["s_rg"] - c["s_in"]) > 12
        inner_low_sat = c["s_in"] < 60
        strong_contrast = c["ring_diff"] > 22
        return (neut_inner and inner_low_sat and ring_more_sat) or (strong_contrast and ring_more_sat)

    ones = [c for c in feats if is_one(c)]
    rest = [c for c in feats if c not in ones]
    for c in ones: out.append((c, 1.00))
    if not rest: return out

    # prateadas
    silver, not_silver = [], []
    for c in rest:
        chroma = np.hypot(c["a"]-128.0, c["b"]-128.0)
        low_color = (chroma < 28) or (c["s"] < 70 and c["v"] > 80)
        (silver if low_color else not_silver).append(c)
    if len(silver) >= 2:
        silver_sorted = sorted(silver, key=lambda x: x["r"])
        out.append((silver_sorted[0], 0.10))
        for c in silver_sorted[1:]: out.append((c, 0.50))
    elif len(silver) == 1:
        c = silver[0]; med_r = np.median([x["r"] for x in rest])
        out.append((c, 0.10 if c["r"] < 0.90*med_r else 0.50))

    rest2 = not_silver
    if not rest2: return out

    # cobre 5c
    copper, brass = [], []
    for c in rest2:
        is_orange = (5 <= c["h"] <= 22)
        lab_red   = (c["a"]-128) > 10 and (c["a"]-c["b"]) > 6
        sat_ok    = c["s"] > 45
        if (is_orange and lab_red) or (lab_red and c["s"] > 60):
            copper.append(c)
        else:
            brass.append(c)
    for c in copper: out.append((c, 0.05))

    # douradas 10c/25c
    if brass:
        if len(brass) == 1:
            c = brass[0]
            out.append((c, 0.25 if c["r"] > np.median([x["r"] for x in feats])*0.95 else 0.10))
        else:
            brass_sorted = sorted(brass, key=lambda x: x["r"])
            out.append((brass_sorted[0], 0.10))
            for c in brass_sorted[1:]: out.append((c, 0.25))
    return out

def annotate_and_sum(img_bgr):
    circles = detect_coins(img_bgr)
    feats = [coin_features(img_bgr, x, y, r) for (x, y, r) in circles]
    classified = classify_all(feats)
    total = 0.0
    out = img_bgr.copy()
    for c, val in classified:
        x, y, r = c["x"], c["y"], c["r"]
        total += val
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        tag = f"R${val:.2f}"
        cv2.putText(out, tag, (x - r, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, tag, (x - r, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (10,10), (320,65), (0,0,0), -1)
    cv2.putText(out, f"TOTAL:  R${total:.2f}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2, cv2.LINE_AA)
    return out, total

# ===================== FLASK APP =====================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = "results"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

PAIR_TOKEN = secrets.token_hex(16)  # troca a cada execução

INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coinsum · Captura</title>
<style>
 body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
 .card { max-width: 720px; margin: auto; }
 .btn  { background:#111; color:#fff; padding:12px 16px; border-radius:10px; border:0; font-size:16px; }
 .row  { margin: 12px 0; }
 .preview { max-width:100%; border:1px solid #ccc; border-radius:8px; }
 .total { font-size:20px; font-weight:700; margin:12px 0; }
 input[type=file] { font-size:16px; }
</style>
</head>
<body>
<div class="card">
  <h2>Contador — Upload da Foto</h2>
  <p>Tire uma foto das moedas (fundo claro, luz difusa) e envie.</p>
  <form method="POST" action="/upload?t={{token}}" enctype="multipart/form-data">
    <div class="row">
      <input type="file" name="photo" accept="image/*" capture="environment" required>
    </div>
    <div class="row">
      <button class="btn" type="submit">Enviar & Processar</button>
    </div>
  </form>
  {% if result_url %}
  <div class="total">TOTAL: R${{ total }}</div>
  <img class="preview" src="{{ result_url }}" alt="Resultado">
  {% endif %}
</div>
</body>
</html>
"""

@app.route("/")
def home():
    t = request.args.get("t", "")
    if t != PAIR_TOKEN: abort(403)
    # Página inicial sem resultado
    return render_template_string(INDEX_HTML, token=PAIR_TOKEN, result_url=None, total=None)

@app.route("/upload", methods=["POST"])
def upload():
    t = request.args.get("t", "")
    if t != PAIR_TOKEN: abort(403)
    if "photo" not in request.files: abort(400)
    f = request.files["photo"]
    if f.filename == "": abort(400)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    up_path = os.path.join(app.config["UPLOAD_FOLDER"], f"shot_{ts}.jpg")
    f.save(up_path)

    img = cv2.imread(up_path)
    if img is None: abort(415)

    out, total = annotate_and_sum(img)

    res_path = os.path.join(app.config["RESULT_FOLDER"], f"result_{ts}.jpg")
    cv2.imwrite(res_path, out)

    # Volta a página com a imagem e total
    rel_url = url_for("get_result", filename=os.path.basename(res_path), _external=False)
    return render_template_string(INDEX_HTML, token=PAIR_TOKEN, result_url=rel_url, total=f"{total:.2f}")

@app.route("/results/<filename>")
def get_result(filename):
    p = os.path.join(app.config["RESULT_FOLDER"], filename)
    if not os.path.exists(p): abort(404)
    return send_file(p, mimetype="image/jpeg")

if __name__ == "__main__":
    ip = get_local_ip()
    url = f"http://{ip}:5000/?t={PAIR_TOKEN}"
    # Gera QR para pareamento
    img = qrcode.make(url)
    img.save("pairing_qr.png")
    print(f"\n📡 Servindo em {url}")
    print("🔐 Token:", PAIR_TOKEN)
    print("📷 Abra o arquivo 'pairing_qr.png' e escaneie com o celular.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
