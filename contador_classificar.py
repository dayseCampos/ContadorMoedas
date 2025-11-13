from sklearn.cluster import KMeans
import cv2
import numpy as np

# ---------- PARÂMETROS-BASE ----------
BASE_REF_WIDTH = 1107
BASE_MIN_R, BASE_MAX_R = 36, 140
BASE_MIN_DIST = 65
BASE_P2 = 32
P1 = 110

# ---------- PRÉ-PROCESSO ----------
def grayworld_wb(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    k = (mb + mg + mr) / 3.0
    b *= k/(mb+1e-6); g *= k/(mg+1e-6); r *= k/(mr+1e-6)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_bgr(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

# ---------- MÁSCARAS ----------
def ring_mask(shape, cx, cy, r, inner=0.70, outer=0.98):
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return (d >= r*inner) & (d <= r*outer)

def circle_mask(shape, cx, cy, r, upto=1.0):
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    d2 = (X - cx)**2 + (Y - cy)**2
    return d2 <= int(r*upto)**2

# ---------- SCORE DE BORDA ----------
def circle_edge_score(img_bgr, x, y, r):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = ring_mask(gray.shape, x, y, r, 0.70, 0.98)
    vals = mag[m]
    return float(vals.mean()) if vals.size else 0.0

# ---------- DETECÇÃO ----------
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

    best_per_cluster = {}
    for (x, y, r, s), lbl in zip(dets, labels):
        cur = best_per_cluster.get(lbl)
        if cur is None or s > cur[3]:
            best_per_cluster[lbl] = (x, y, r, s)

    picked = list(best_per_cluster.values())
    picked.sort(key=lambda t: -t[3])
    return [(x, y, r) for (x, y, r, _) in picked]

# ---------- FEATURES ----------
def coin_features(img_bgr, x, y, r):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    A, B = lab[:, :, 1], lab[:, :, 2]

    h, w = img_bgr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    d2 = (X - x) ** 2 + (Y - y) ** 2
    m_coin  = d2 <= r * r
    m_inner = d2 <= int(0.58 * r) ** 2
    m_ring  = (d2 <= int(0.98 * r) ** 2) & (d2 >= int(0.70 * r) ** 2)

    def mmean(M, mask):
        vals = M[mask]; return float(vals.mean()) if vals.size else 0.0

    h_coin, s_coin, v_coin = mmean(H, m_coin), mmean(S, m_coin), mmean(V, m_coin)
    h_in, s_in, v_in = mmean(H, m_inner), mmean(S, m_inner), mmean(V, m_inner)
    h_rg, s_rg, v_rg = mmean(H, m_ring),  mmean(S, m_ring),  mmean(V, m_ring)
    a_coin, b_coin = mmean(A, m_coin), mmean(B, m_coin)
    ring_diff = np.linalg.norm([h_in - h_rg, s_in - s_rg, v_in - v_rg])

    return {
        "x": x, "y": y, "r": r,
        "h": h_coin, "s": s_coin, "v": v_coin,
        "h_in": h_in, "s_in": s_in, "v_in": v_in,
        "h_rg": h_rg, "s_rg": s_rg, "v_rg": v_rg,
        "a": a_coin, "b": b_coin,
        "ring_diff": ring_diff
    }

# ---------- CLASSIFICAÇÃO ----------
def classify_all(feats):
    out = []

    # --- R$1,00 bimetálica ---
    def is_one(c):
        neut_inner = abs(c["a"] - 128) + abs(c["b"] - 128) < 22
        ring_more_sat = (c["s_rg"] - c["s_in"]) > 12
        inner_low_sat = c["s_in"] < 60
        strong_contrast = c["ring_diff"] > 22
        return (neut_inner and inner_low_sat and ring_more_sat) or (strong_contrast and ring_more_sat)

    ones = [c for c in feats if is_one(c)]
    rest = [c for c in feats if c not in ones]
    for c in ones:
        out.append((c, 1.00))
    if not rest:
        return out

    # --- prateadas (0,50 e 0,10 antiga) ---
    silver, not_silver = [], []
    for c in rest:
        chroma = np.hypot(c["a"] - 128.0, c["b"] - 128.0)
        low_color = chroma < 28 or (c["s"] < 70 and c["v"] > 80)
        (silver if low_color else not_silver).append(c)

    if len(silver) >= 2:
        silver_sorted = sorted(silver, key=lambda x: x["r"])
        out.append((silver_sorted[0], 0.10))
        for c in silver_sorted[1:]:
            out.append((c, 0.50))
    elif len(silver) == 1:
        c = silver[0]
        median_r = np.median([x["r"] for x in rest])
        out.append((c, 0.10 if c["r"] < 0.90 * median_r else 0.50))

    rest2 = not_silver

    if not rest2:
        return out

    # --- cobre 5c (alaranjado/vermelho) vs. douradas (10c/25c) ---
    copper, brass = [], []
    for c in rest2:
        h = c["h"]
        s = c["s"]
        a_off = c["a"] - 128
        b_off = c["b"] - 128

        is_copper = (
            h < 20 and
            s > 60 and
            (a_off - b_off) > 5
        )

        if is_copper:
            copper.append(c)
        else:
            brass.append(c)

    for c in copper:
        out.append((c, 0.05))

    # se não detectou cobre, tenta achar o mais avermelhado
    if not copper and brass:
        cand = min(brass, key=lambda c: (c["h"] + 0.5*max(0, (c["b"]-128))))
        if cand["h"] < 22 and cand["s"] > 55 and (cand["a"] - cand["b"]) > 3:
            brass.remove(cand)
            out.append((cand, 0.05))

    # --- douradas (0,10 e 0,25) por raio ---
    if brass:
        if len(brass) == 1:
            c = brass[0]
            med_r = np.median([x["r"] for x in feats])
            out.append((c, 0.25 if c["r"] > 0.95 * med_r else 0.10))
        else:
            brass_sorted = sorted(brass, key=lambda x: x["r"])
            out.append((brass_sorted[0], 0.10))
            for c in brass_sorted[1:]:
                out.append((c, 0.25))

    return out

# ---------- MAIN ----------
def main():
    img = cv2.imread("moedas.jpg")
    if img is None:
        print("❌ Coloque a imagem 'moedas.jpg' na pasta.")
        return

    circles = detect_coins(img)
    feats = [coin_features(img, x, y, r) for (x, y, r) in circles]
    classified = classify_all(feats)

    total = 0.0
    out = img.copy()
    for c, val in classified:
        x, y, r = c["x"], c["y"], c["r"]
        total += val
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        tag = f"R${val:.2f}"
        cv2.putText(out, tag, (x - r, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, tag, (x - r, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(out, (10, 10), (320, 65), (0, 0, 0), -1)
    cv2.putText(out, f"TOTAL:  R${total:.2f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite("resultado.jpg", out)
    print(f"\n💰 Total detectado: R${total:.2f}")
    print("💾 Resultado salvo em 'resultado.jpg'.")

if __name__ == "__main__":
    main()
