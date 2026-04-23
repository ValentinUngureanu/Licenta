"""
LEDD.py — Lung Disease Diagnosis
---------------------------------
Pipeline de detectie a liniei pleurale pe ecografii transthoracice (TUS).

Pipeline:
    1. PixelConverter   → calculeaza rezolutia (mm/pixel)
    2. CropBorder       → elimina borderul si UI-ul ecografului
    3. remove_noise     → bilateral filter + CLAHE
    4. ExtractPleuralLine → detectie pleura prin scanline A-mode + umbra acustica

Autor: Vali (licenta)
"""

import re

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tes
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, medfilt
from skimage import io
from skimage.color import rgb2gray


# ═════════════════════════════════════════════════════════════
#  PARAMETRI EXTRACT PLEURAL LINE (ajustabili)
# ═════════════════════════════════════════════════════════════

# Zona de cautare a pleurei, in mm de la partea de sus a imaginii cropate
DEPTH_MIN_MM = 15.0       # pleura e rareori mai sus de 1.5 cm
DEPTH_MAX_MM = 40.0       # limita superioara — empirica, din histograma

# Fereastra de analiza pentru umbra acustica (in mm, sub candidat)
SHADOW_OFFSET_MM = 3.0    # buffer imediat dedesubt (skip zona de reverberatie)
SHADOW_WINDOW_MM = 15.0   # cat de mult masuram dedesubt

# Filtrare varfuri
PEAK_PROMINENCE = 0.08    # prominenta minima (pe scala [0,1])
PEAK_MIN_DISTANCE_PX = 8  # distanta minima intre varfuri

# RANSAC (folosit doar ca fallback daca DP esueaza)
RANSAC_POLY_DEGREE = 2
RANSAC_THRESH_PX = 10
RANSAC_ITERATIONS = 300
RANSAC_MIN_INLIERS_FRAC = 0.3

# Dynamic Programming (metoda principala de detectie)
DP_LAMBDA = 0.3          # penalizare pt salt vertical (mai mare = linie mai "neteda")
DP_MAX_JUMP_PX = 25      # saltul maxim permis intre coloane vecine (px)
DP_MIN_SCORE_THRESH = 0.05  # scor minim pt o coloana sa fie "valida" dupa DP
DP_MIN_VALID_FRAC = 0.4  # % minim coloane valide pt acceptarea caii DP

# Contur pleura (extragere margini sus/jos + segmentare pe intreruperi)
INTERRUPT_MIN_GAP_MM = 3.0   # gap ≥ 3mm = intrerupere reala (sub asta = zgomot)
BAND_INTENSITY_FRAC = 0.75   # extinde banda cat timp I >= 75% din I(varf) — mai strict
BAND_MAX_HALF_WIDTH_MM = 1.0 # grosime max jumatate banda (total max 2mm — fizic realist)


# ═════════════════════════════════════════════════════════════
#  1. CROP BORDER
# ═════════════════════════════════════════════════════════════

def CropBorder(orig_Image):
    """
    Elimina borderul imaginii ecografice si UI-ul (bare laterale, topbar).
    Returneaza imaginea cropata in grayscale.
    """
    orig_img = orig_Image.copy()
    gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

    _, img_bin = cv2.threshold(gray_Image, 128, 255,
                                cv2.THRESH_BINARY | cv2.ADAPTIVE_THRESH_MEAN_C)
    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Umple poligoanele mari (≥1000 px²) pe imaginea originala
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    # Reface threshold + contururi
    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Deseneaza aceleasi poligoane pe o imagine neagra (pt boundary detection)
    black = np.zeros_like(img_bin)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) in (2, 4):
                cv2.drawContours(black, [approx], 0, 255, -1)

    # Left bound
    white_pixels = np.array(np.where(black == 255))
    last_small = (white_pixels[1, white_pixels[1] < 100]
                  if len(white_pixels[1]) > 0 else np.array([]))

    if len(white_pixels[1]) == 0 or len(last_small) == 0:
        left_bound = 25
    else:
        left_bound = last_small[-1]

    # Refacere pentru poligoane mici (ticks pe bara laterala)
    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    black = np.zeros_like(img_bin)
    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)
            if len(approx) in (2, 4):
                cv2.drawContours(black, [approx], 0, 255, -1)

    # Detectie bara verticala (coloana cu cele mai multe tick-uri)
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    img_temp2 = cv2.erode(black, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    bar_image = horizontal_lines_img
    columns = np.zeros(bar_image.shape[1], dtype=int)
    for i in range(bar_image.shape[1]):
        columns[i] = np.count_nonzero(bar_image[:, i])

    bar_pos = np.where(columns == np.max(columns))[0][0]
    bar = bar_image[:, bar_pos] // 255
    bar_pixels = np.array(np.where(bar == 1))
    first_bar_pixel = bar_pixels[:, 0]
    last_bar_pixel = bar_pixels[:, -1]

    # Crop
    if first_bar_pixel[0] == 0 or last_bar_pixel[0] == 0:
        print("[CropBorder] Eroare: imagine neconforma, nu s-a cropat")
        return gray_Image.copy()

    if len(white_pixels[1]) == 0 or len(last_small) == 0:
        print("[CropBorder] Avertisment: bara intensitate negasita, crop aproximativ")

    crop_img = gray_Image[first_bar_pixel[0]:last_bar_pixel[0],
                          left_bound:bar_pos - 20].copy()
    return crop_img


# ═════════════════════════════════════════════════════════════
#  2. REMOVE NOISE
# ═════════════════════════════════════════════════════════════

def remove_noise(img):
    """Bilateral filter + CLAHE. Returneaza grayscale uint8."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


# ═════════════════════════════════════════════════════════════
#  3. PIXEL CONVERTER (mm per pixel)
# ═════════════════════════════════════════════════════════════

def PixelConverter(orig_Image):
    """
    Calculeaza rezolutia (mm/pixel) folosind OCR pe adancimea afisata
    si detectia tick-urilor de pe bara verticala de adancime.
    """
    orig_img = orig_Image.copy()
    # BGR2GRAY — pastrat exact ca in versiunea originala (calibrata pe asta)
    gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    _, img_bin = cv2.threshold(gray_Image, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Deseneaza poligoane mici pe imaginea neagra
    black = np.zeros_like(img_bin)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)
            if len(approx) in (2, 4):
                cv2.drawContours(black, [approx], 0, 255, -1)

    # Detectie bara verticala
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    img_temp2 = cv2.erode(black, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    columns = np.zeros(horizontal_lines_img.shape[1], dtype=int)
    for i in range(horizontal_lines_img.shape[1]):
        columns[i] = np.count_nonzero(horizontal_lines_img[:, i])

    bar_pos = np.where(columns == np.max(columns))[0][0]
    bar = horizontal_lines_img[:, bar_pos] // 255
    indices = [i for i, x in enumerate(bar) if x == 1]

    print(f"[PixelConverter] Tick-uri detectate pe bara: {len(indices)}")
    if len(indices) >= 2:
        tick_spacing = np.sum(np.diff(indices))
        print(f"[PixelConverter] Spatiu total tick-uri: {tick_spacing} px")
    else:
        print(f"[PixelConverter] EROARE: Prea putine tick-uri pe bara!")
        return 0.07  # fallback tipic pt ecografia ta

    # OCR pentru adancime
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image_str = tes.image_to_string(np.invert(img_bin))

    print(f"[PixelConverter] OCR text (primele 200 char):")
    print(f"  {repr(image_str[:200])}")

    depth = None

    # Prioritate 1: formatul standard '-D 45' sau '\nD 45' (adancime ecograf)
    # Acestea sunt cele mai fiabile — formatul specific al ecografelor GE/Philips/etc.
    for token in ('-D ', '\nD '):
        if token in image_str:
            i0 = image_str.find(token) + len(token)
            # Extrage primul numar INTREG dupa token (nu cifre individuale)
            remaining = image_str[i0:i0 + 10]
            match = re.search(r'\d+', remaining)
            if match:
                val = int(match.group())
                # Heuristic: valori 15-200 sunt plauzibile ca mm (adancime ecograf)
                # Valori 1-14 sunt probabil cm (converteste la mm)
                if val <= 14:
                    depth = val * 10  # cm → mm
                else:
                    depth = val  # deja mm
                print(f"[PixelConverter] Adancime gasita prin '{token.strip()}': {depth} mm (val citit: {val})")
                break

    # Prioritate 2: 'cm' / 'mm' dar DOAR daca nu e precedat de '.' sau 'L '
    # (evita prinderea '1L 0.56 cm' care e lungime masurata, nu adancime)
    if depth is None and 'cm' in image_str:
        idx_cm = image_str.find('cm')
        before = image_str[max(0, idx_cm - 8):idx_cm]
        if '.' not in before and 'L ' not in before:
            depth_no = [int(s) for s in re.findall(r'\b\d+\b', before)]
            if depth_no:
                depth = depth_no[0] * 10
                print(f"[PixelConverter] Adancime gasita prin 'cm': {depth} mm")

    if depth is None and 'mm' in image_str:
        idx_mm = image_str.find('mm')
        before = image_str[max(0, idx_mm - 8):idx_mm]
        if '.' not in before and 'L ' not in before:
            depth_no = [int(s) for s in re.findall(r'\b\d+\b', before)]
            if depth_no:
                depth = depth_no[0]
                print(f"[PixelConverter] Adancime gasita prin 'mm': {depth} mm")

    if depth is None:
        print('[PixelConverter] AVERTISMENT: adancime negasita prin OCR!')
        print('[PixelConverter] Introdu manual adancimea (mm) vizibila pe ecografie,')
        print('[PixelConverter] sau seteaza one_pixel hardcodat in main().')
        # Fallback: asuma 40mm (tipic pt ecografie pulmonara)
        depth = 40
        print(f'[PixelConverter] Folosesc fallback: depth = {depth} mm')

    tick_sum = np.sum(np.diff(indices))
    if tick_sum == 0:
        print(f"[PixelConverter] EROARE: sum(diff(indices)) = 0, nu pot diviza!")
        return 0.07

    depth_pix = depth / tick_sum
    print(f"[PixelConverter] Rezultat: {depth} mm / {tick_sum} px = {depth_pix:.4f} mm/px")
    return depth_pix


# ═════════════════════════════════════════════════════════════
#  4. EXTRACT PLEURAL LINE — helpers
# ═════════════════════════════════════════════════════════════

def _to_gray_float(img):
    """Converteste la grayscale float32 in [0, 1], indiferent de input."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        gray = rgb2gray(img)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    return np.clip(gray, 0.0, 1.0)


def _apply_clahe(gray_float):
    """CLAHE pe grayscale float [0,1]. Returneaza tot float [0,1]."""
    gray_u8 = (gray_float * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_u8 = clahe.apply(gray_u8)
    return enhanced_u8.astype(np.float32) / 255.0


def _mm_to_px(mm, one_pixel_mm):
    """Converteste milimetri in pixeli."""
    return int(round(mm / one_pixel_mm))


def _score_column(column, y_min, y_max, shadow_offset_px, shadow_window_px):
    """
    Analizeaza o coloana si intoarce (y_best, score_best).
    Scor = intensitate(varf) - medie(fereastra_dedesubt).
    """
    H = len(column)
    search_zone = column[y_min:y_max]
    if len(search_zone) < 5:
        return None, -np.inf

    peaks_local, _ = find_peaks(
        search_zone,
        prominence=PEAK_PROMINENCE,
        distance=PEAK_MIN_DISTANCE_PX,
    )
    if len(peaks_local) == 0:
        return None, -np.inf

    peaks_global = peaks_local + y_min
    best_y, best_score = None, -np.inf

    for y_peak in peaks_global:
        y_start = y_peak + shadow_offset_px
        y_end = y_peak + shadow_offset_px + shadow_window_px
        if y_end >= H:
            continue

        mu_below = column[y_start:y_end].mean()
        score = column[y_peak] - mu_below
        if score > best_score:
            best_score = score
            best_y = y_peak

    return best_y, best_score


def _ransac_poly(xs, ys, degree=2, thresh=15, iterations=200):
    """RANSAC pentru fit polinomial. Returneaza (coefs, inlier_mask) sau (None, None)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    if n < degree + 1:
        return None, None

    best_inliers, best_coefs, best_count = None, None, 0
    sample_size = degree + 1
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(n, sample_size, replace=False)
        try:
            coefs = np.polyfit(xs[idx], ys[idx], degree)
        except np.linalg.LinAlgError:
            continue

        residuals = np.abs(ys - np.polyval(coefs, xs))
        inliers = residuals < thresh
        count = inliers.sum()
        if count > best_count:
            best_count, best_inliers, best_coefs = count, inliers, coefs

    if best_inliers is None or best_count < sample_size:
        return None, None

    try:
        final_coefs = np.polyfit(xs[best_inliers], ys[best_inliers], degree)
        return final_coefs, best_inliers
    except np.linalg.LinAlgError:
        return best_coefs, best_inliers


# ─────────────────────────────────────────────────────────────
#  DYNAMIC PROGRAMMING — cale optima prin imagine
# ─────────────────────────────────────────────────────────────

def _compute_cost_map(enhanced, y_min, y_max, shadow_offset_px, shadow_window_px):
    """
    Calculeaza hartea de cost pt DP. Pentru fiecare (x, y) in zona de cautare:
      score(x, y) = I(x, y) - mean(I(x, y+offset : y+offset+window))
      cost(x, y) = 1 - score_normalizat (cu cat e mai mare scorul, cu atat cost mai mic)

    Returneaza:
      cost_map  : np.array(H, W) — costuri in [0, 1], inf in zona invalida
      score_map : np.array(H, W) — scorurile brute (pt raportare)
    """
    H, W = enhanced.shape
    score_map = np.full((H, W), -np.inf, dtype=np.float32)

    # Precalculeaza suma cumulativa pe coloane pt fereastra medie rapida
    # csum[i, x] = suma enhanced[0:i, x]
    csum = np.concatenate([np.zeros((1, W), dtype=np.float32),
                            np.cumsum(enhanced, axis=0, dtype=np.float32)], axis=0)

    # Pentru fiecare y din zona valida, calculeaza scorul simultan pe toate coloanele
    for y in range(y_min, y_max + 1):
        y_start = y + shadow_offset_px
        y_end = y + shadow_offset_px + shadow_window_px
        if y_end >= H:
            continue

        # Intensitatea la (y, all_x)
        i_peak = enhanced[y, :]
        # Media ferestrei de umbra pe fiecare coloana
        window_sum = csum[y_end, :] - csum[y_start, :]
        window_mean = window_sum / shadow_window_px

        score_map[y, :] = i_peak - window_mean

    # Transforma scor → cost. Valori mari scor = cost mic. Norm la [0, 1].
    valid = score_map > -np.inf
    if not valid.any():
        return np.full_like(score_map, np.inf), score_map

    s_min = score_map[valid].min()
    s_max = score_map[valid].max()
    s_range = max(s_max - s_min, 1e-6)

    cost_map = np.full_like(score_map, np.inf, dtype=np.float32)
    cost_map[valid] = 1.0 - (score_map[valid] - s_min) / s_range

    return cost_map, score_map


def _dp_optimal_path(cost_map, lambda_penalty, max_jump):
    """
    Dynamic Programming: gaseste calea optima (un y per coloana) minimizand:
      sum_x [ cost(x, y_x) ] + lambda * sum_x [ |y_x - y_{x-1}| ]

    Saltul vertical intre coloane vecine e limitat la max_jump px.

    Returneaza:
      path    : np.array(W,) coordonate y optime per coloana
      dp_cost : np.array(W,) costul cumulativ pe calea optima (pt evaluare)
    """
    H, W = cost_map.shape

    # Tabela DP: dp[y, x] = cost minim al caii optime pana la (x, y)
    # NOTA: pastram aceeasi conventie ca cost_map (H, W) pentru usurinta indexarii
    dp = np.full((H, W), np.inf, dtype=np.float32)
    # Parent pointer: pt a reconstrui calea
    parent = np.full((H, W), -1, dtype=np.int32)

    # Prima coloana: cost initial = cost_local
    dp[:, 0] = cost_map[:, 0]

    # Forward pass
    for x in range(1, W):
        for y in range(H):
            if cost_map[y, x] == np.inf:
                continue

            # Limite y_prev valide
            y_lo = max(0, y - max_jump)
            y_hi = min(H, y + max_jump + 1)

            # Cost candidat pt fiecare y_prev
            prev_costs = dp[y_lo:y_hi, x - 1]
            transition = lambda_penalty * np.abs(np.arange(y_lo, y_hi) - y)
            total = prev_costs + transition

            best_idx = np.argmin(total)
            best_cost = total[best_idx]

            if best_cost < np.inf:
                dp[y, x] = cost_map[y, x] + best_cost
                parent[y, x] = y_lo + best_idx

    # Backward pass: reconstruieste calea optima
    path = np.zeros(W, dtype=np.int32)

    # Ultima coloana: aleg y cu cost minim
    last_col_costs = dp[:, W - 1]
    if np.all(np.isinf(last_col_costs)):
        return None, None
    path[W - 1] = int(np.argmin(last_col_costs))

    # Urmareste parent pointers inapoi
    for x in range(W - 1, 0, -1):
        path[x - 1] = parent[path[x], x]
        if path[x - 1] < 0:
            break

    dp_cost = np.array([dp[path[x], x] for x in range(W)])

    return path, dp_cost


def _validate_dp_path(path, score_map, min_score_thresh):
    """
    Marcheaza ca "valide" doar coloanele unde scorul de umbra acustica
    la pozitia aleasa de DP e peste un prag. Restul devin NaN.

    Returneaza:
      pleura_y_dp  : np.array(W,) float (NaN unde invalid)
      valid_mask   : np.array(W,) bool
      scores_along_path : np.array(W,) scorul la fiecare pozitie
    """
    W = len(path)
    scores_along_path = np.array([score_map[path[x], x] for x in range(W)], dtype=np.float32)

    valid_mask = scores_along_path >= min_score_thresh
    pleura_y_dp = np.where(valid_mask, path.astype(np.float64), np.nan)

    return pleura_y_dp, valid_mask, scores_along_path


# ═════════════════════════════════════════════════════════════
#  4. EXTRACT PLEURAL LINE — functia principala
# ═════════════════════════════════════════════════════════════

def ExtractPleuralLine(crop_image, one_pixel_mm, debug=False, debug_columns=None,
                        method='dp'):
    """
    Detecteaza linia pleurala.

    Metode disponibile:
      - 'dp'     : Dynamic Programming (RECOMANDAT) — gaseste calea optima globala
      - 'legacy' : scanline + RANSAC (metoda veche, pastrata ca fallback)

    Returneaza dict cu:
        pleura_y, pleura_y_smooth, contour, scores, inlier_mask,
        confidence, enhanced, method (ce metoda s-a folosit)
    """
    # ─ 1. Preprocessing (la fel ca inainte)
    gray = _to_gray_float(crop_image)
    enhanced = _apply_clahe(gray)
    H, W = enhanced.shape

    if one_pixel_mm <= 0:
        raise ValueError(
            f"one_pixel_mm invalid: {one_pixel_mm}. "
            f"PixelConverter a esuat — hardcodeaza o valoare (tipic 0.05-0.10 mm/px)."
        )

    y_min = max(0, _mm_to_px(DEPTH_MIN_MM, one_pixel_mm))
    y_max = min(H - 1, _mm_to_px(DEPTH_MAX_MM, one_pixel_mm))
    if y_max <= y_min + 10:
        raise ValueError(f"Zona cautare prea mica: [{y_min}, {y_max}]")

    shadow_offset_px = max(2, _mm_to_px(SHADOW_OFFSET_MM, one_pixel_mm))
    shadow_window_px = max(5, _mm_to_px(SHADOW_WINDOW_MM, one_pixel_mm))

    method_used = method

    if method == 'dp':
        # ─ 2a. Metoda DP
        print("[ExtractPleuralLine] Rulez Dynamic Programming...")
        cost_map, score_map = _compute_cost_map(
            enhanced, y_min, y_max, shadow_offset_px, shadow_window_px
        )

        path, dp_cost = _dp_optimal_path(
            cost_map, lambda_penalty=DP_LAMBDA, max_jump=DP_MAX_JUMP_PX
        )

        if path is None:
            print("[ExtractPleuralLine] DP a esuat, folosesc metoda legacy.")
            method_used = 'legacy'
        else:
            pleura_y, valid_mask, scores_path = _validate_dp_path(
                path, score_map, DP_MIN_SCORE_THRESH
            )

            frac_valid = valid_mask.sum() / W
            print(f"[ExtractPleuralLine] DP: {valid_mask.sum()}/{W} coloane valide ({100*frac_valid:.1f}%)")

            if frac_valid < DP_MIN_VALID_FRAC:
                print(f"[ExtractPleuralLine] Prea putine coloane valide (<{100*DP_MIN_VALID_FRAC:.0f}%), incerc legacy.")
                method_used = 'legacy'
            else:
                # DP a mers bine. Nu mai avem nevoie de RANSAC — linia e deja continua.
                scores = scores_path.astype(np.float64)
                scores[~valid_mask] = -np.inf
                inlier_mask = valid_mask.copy()

    if method_used == 'legacy':
        # ─ 2b. Metoda veche (scanline + RANSAC)
        print("[ExtractPleuralLine] Rulez scanline + RANSAC (legacy)...")
        pleura_y = np.full(W, np.nan, dtype=np.float64)
        scores = np.full(W, -np.inf, dtype=np.float64)

        for x in range(W):
            y_best, score_best = _score_column(
                enhanced[:, x], y_min, y_max, shadow_offset_px, shadow_window_px
            )
            if y_best is not None:
                pleura_y[x] = y_best
                scores[x] = score_best

        valid_mask = ~np.isnan(pleura_y)
        xs_valid = np.where(valid_mask)[0]
        ys_valid = pleura_y[valid_mask]

        _, inliers_of_valid = _ransac_poly(
            xs_valid, ys_valid,
            degree=RANSAC_POLY_DEGREE,
            thresh=RANSAC_THRESH_PX,
            iterations=RANSAC_ITERATIONS,
        )

        inlier_mask = np.zeros(W, dtype=bool)
        if inliers_of_valid is not None:
            inlier_mask[xs_valid[inliers_of_valid]] = True
        else:
            inlier_mask = valid_mask.copy()

    # ─ 3. Smoothing + interpolare (la fel pt ambele metode)
    xs_inlier = np.where(inlier_mask)[0]
    ys_inlier = pleura_y[inlier_mask]
    pleura_y_smooth = np.full(W, np.nan)

    if len(xs_inlier) >= 4:
        ys_med = medfilt(ys_inlier, kernel_size=5) if len(ys_inlier) >= 5 else ys_inlier
        try:
            spline = UnivariateSpline(xs_inlier, ys_med, k=3, s=len(xs_inlier))
            x_min_i, x_max_i = xs_inlier.min(), xs_inlier.max()
            x_full = np.arange(x_min_i, x_max_i + 1)
            y_full = spline(x_full)

            y_lo = ys_inlier.min() - 10
            y_hi = ys_inlier.max() + 10
            mask_reasonable = (y_full >= y_lo) & (y_full <= y_hi)

            valid_x = x_full[mask_reasonable]
            valid_y = y_full[mask_reasonable]
            pleura_y_smooth[valid_x] = valid_y
        except Exception as e:
            print(f"[ExtractPleuralLine] Spline esuat ({e}), interpolare liniara.")
            pleura_y_smooth = np.interp(np.arange(W), xs_inlier, ys_inlier,
                                         left=np.nan, right=np.nan)

    # ─ 4. Contur format OpenCV
    valid_smooth = ~np.isnan(pleura_y_smooth)
    xs_c = np.where(valid_smooth)[0]
    ys_c = pleura_y_smooth[valid_smooth].astype(np.int32)
    contour = np.stack([xs_c, ys_c], axis=-1).reshape(-1, 1, 2)

    # ─ 5. Confidence
    if inlier_mask.sum() > 0:
        valid_scores_for_conf = scores[inlier_mask]
        valid_scores_for_conf = valid_scores_for_conf[valid_scores_for_conf > -np.inf]
        if len(valid_scores_for_conf) > 0:
            confidence = float(np.clip(valid_scores_for_conf.mean() / 0.3, 0.0, 1.0))
        else:
            confidence = 0.0
    else:
        confidence = 0.0

    # ─ 6. Debug plot (ca inainte)
    if debug:
        _debug_plot(
            crop_image, enhanced, pleura_y, pleura_y_smooth,
            scores, inlier_mask, y_min, y_max,
            shadow_offset_px, shadow_window_px,
            one_pixel_mm, confidence, debug_columns,
        )

    return {
        'pleura_y': pleura_y,
        'pleura_y_smooth': pleura_y_smooth,
        'contour': contour,
        'scores': scores,
        'inlier_mask': inlier_mask,
        'confidence': confidence,
        'enhanced': enhanced,
        'method': method_used,
    }


# ═════════════════════════════════════════════════════════════
#  5. EXTRACT PLEURAL CONTOUR (margini sus/jos + intreruperi)
# ═════════════════════════════════════════════════════════════

def _find_band_edges(column, y_center, intensity_frac, max_half_width_px):
    """
    Pornind de la y_center pe coloana data, extinde sus si jos cat timp
    intensitatea ramane peste intensity_frac * I(y_center).

    Returneaza (y_top, y_bottom).
    """
    H = len(column)
    y_center = int(y_center)
    if y_center < 0 or y_center >= H:
        return y_center, y_center

    i_peak = column[y_center]
    threshold = intensity_frac * i_peak

    # Extinde in sus
    y_top = y_center
    for dy in range(1, max_half_width_px + 1):
        y = y_center - dy
        if y < 0 or column[y] < threshold:
            break
        y_top = y

    # Extinde in jos
    y_bot = y_center
    for dy in range(1, max_half_width_px + 1):
        y = y_center + dy
        if y >= H or column[y] < threshold:
            break
        y_bot = y

    return y_top, y_bot


def _find_continuous_segments(pleura_smooth, min_gap_px):
    """
    Identifica segmente continue de pleura in pleura_smooth (array cu NaN-uri).
    Un "gap" (secventa de NaN) ≥ min_gap_px separa doua segmente.

    Returneaza lista de tupluri (x_start, x_end) inclusive.
    """
    W = len(pleura_smooth)
    valid = ~np.isnan(pleura_smooth)

    if valid.sum() == 0:
        return []

    # Detecteaza tranzitii valid↔invalid
    # Prefix -1 si sufix -1 ca sa captam inceput/sfarsit
    padded = np.concatenate(([False], valid, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]   # unde incepe un segment valid
    ends = np.where(diff == -1)[0] - 1  # unde se termina (inclusive)

    segments_raw = list(zip(starts, ends))

    # Fuzionam segmente cu gap mai mic decat min_gap_px
    if not segments_raw:
        return []

    merged = [segments_raw[0]]
    for s, e in segments_raw[1:]:
        prev_s, prev_e = merged[-1]
        gap = s - prev_e - 1
        if gap < min_gap_px:
            merged[-1] = (prev_s, e)  # extinde segmentul anterior
        else:
            merged.append((s, e))

    return merged


def ExtractPleuralContour(result, img_denoised, one_pixel_mm):
    """
    Construieste contur(uri) complete pentru pleura detectata de ExtractPleuralLine.

    Pentru fiecare segment continuu de pleura (intre intreruperi):
      - Extinde banda sus/jos pe fiecare coloana (margine hiperecogena)
      - Construieste contur inchis: margine_sus stanga→dreapta, margine_jos dreapta→stanga

    Parametri:
      result          : dict returnat de ExtractPleuralLine
      img_denoised    : imaginea uint8 (dupa remove_noise) — necesara pt intensitati
      one_pixel_mm    : rezolutia in mm/px

    Returneaza dict:
      {
        'segments'      : lista de tupluri (x_start, x_end) — segmente continue,
        'interruptions' : lista de tupluri (x_start, x_end, width_mm) — gap-urile,
        'top_edges'     : lista np.array(shape=(n,2)) cu puncte (x,y) margine sus per segment,
        'bottom_edges'  : lista np.array(shape=(n,2)) cu puncte (x,y) margine jos per segment,
        'contours_cv'   : lista de contururi format OpenCV (N,1,2) — cate unul per segment,
        'thickness_mm'  : np.array(W,) grosimea pleurei per coloana (NaN unde nu exista)
      }
    """
    pleura_smooth = result['pleura_y_smooth']
    enhanced = result['enhanced']  # float [0,1] — avem intensitati corecte
    H, W = enhanced.shape

    min_gap_px = max(1, _mm_to_px(INTERRUPT_MIN_GAP_MM, one_pixel_mm))
    max_half_width_px = max(2, _mm_to_px(BAND_MAX_HALF_WIDTH_MM, one_pixel_mm))

    # ─ 1. Identifica segmente continue si intreruperi
    segments = _find_continuous_segments(pleura_smooth, min_gap_px)

    # Intreruperile = gap-urile intre segmente + capetele imaginii
    interruptions = []
    if segments:
        # Gap initial (daca primul segment nu incepe la x=0)
        if segments[0][0] > 0:
            w_mm = segments[0][0] * one_pixel_mm
            if w_mm >= INTERRUPT_MIN_GAP_MM:
                interruptions.append((0, segments[0][0] - 1, w_mm))

        # Gap-uri intre segmente
        for i in range(len(segments) - 1):
            x_end_prev = segments[i][1]
            x_start_next = segments[i + 1][0]
            gap_start = x_end_prev + 1
            gap_end = x_start_next - 1
            w_mm = (gap_end - gap_start + 1) * one_pixel_mm
            interruptions.append((gap_start, gap_end, w_mm))

        # Gap final
        if segments[-1][1] < W - 1:
            gap_start = segments[-1][1] + 1
            gap_end = W - 1
            w_mm = (gap_end - gap_start + 1) * one_pixel_mm
            if w_mm >= INTERRUPT_MIN_GAP_MM:
                interruptions.append((gap_start, gap_end, w_mm))

    # ─ 2. Pentru fiecare segment, construieste margini sus/jos
    top_edges = []
    bottom_edges = []
    contours_cv = []
    thickness_mm = np.full(W, np.nan)

    for x_start, x_end in segments:
        top_pts = []
        bot_pts = []

        for x in range(x_start, x_end + 1):
            y_c = pleura_smooth[x]
            if np.isnan(y_c):
                continue

            column = enhanced[:, x]
            y_top, y_bot = _find_band_edges(
                column, y_c, BAND_INTENSITY_FRAC, max_half_width_px
            )

            top_pts.append([x, y_top])
            bot_pts.append([x, y_bot])
            thickness_mm[x] = (y_bot - y_top) * one_pixel_mm

        if len(top_pts) < 2:
            continue  # segment prea scurt

        top_arr = np.array(top_pts, dtype=np.int32)
        bot_arr = np.array(bot_pts, dtype=np.int32)

        top_edges.append(top_arr)
        bottom_edges.append(bot_arr)

        # Contur inchis OpenCV: sus stanga→dreapta + jos dreapta→stanga
        cnt = np.concatenate([top_arr, bot_arr[::-1]], axis=0)
        cnt_cv = cnt.reshape(-1, 1, 2)
        contours_cv.append(cnt_cv)

    # ─ 3. Statistici (pentru logging)
    valid_th = ~np.isnan(thickness_mm)
    print(f"\n[ExtractPleuralContour]")
    print(f"  Segmente continue: {len(segments)}")
    print(f"  Intreruperi: {len(interruptions)}")
    for i, (xs, xe, wmm) in enumerate(interruptions):
        print(f"    #{i+1}: x={xs}-{xe} ({wmm:.1f} mm)")
    if valid_th.sum() > 0:
        print(f"  Grosime pleura — mean: {thickness_mm[valid_th].mean():.2f} mm, "
              f"max: {thickness_mm[valid_th].max():.2f} mm, "
              f"min: {thickness_mm[valid_th].min():.2f} mm")

    return {
        'segments': segments,
        'interruptions': interruptions,
        'top_edges': top_edges,
        'bottom_edges': bottom_edges,
        'contours_cv': contours_cv,
        'thickness_mm': thickness_mm,
    }


# ═════════════════════════════════════════════════════════════
#  DEBUG VISUALIZATION
# ═════════════════════════════════════════════════════════════

def _debug_plot(crop_image, enhanced, pleura_y, pleura_y_smooth,
                scores, inlier_mask, y_min, y_max,
                shadow_offset_px, shadow_window_px,
                one_pixel_mm, confidence, debug_columns=None):
    """Grid de diagnostic: imagine + CLAHE + profile pe 5 coloane."""
    H, W = enhanced.shape
    if debug_columns is None:
        debug_columns = np.linspace(int(W * 0.15), int(W * 0.85), 5, dtype=int).tolist()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, len(debug_columns), hspace=0.35, wspace=0.3)

    # Rand 1: imagine originala + detectii
    ax_orig = fig.add_subplot(gs[0, :])
    if crop_image.ndim == 3:
        ax_orig.imshow(crop_image)
    else:
        ax_orig.imshow(crop_image, cmap='gray')

    xs_det = np.where(~np.isnan(pleura_y))[0]
    ax_orig.scatter(xs_det, pleura_y[xs_det], s=2, c='red', alpha=0.4,
                    label='Detectii brute')

    xs_in = np.where(inlier_mask)[0]
    ax_orig.scatter(xs_in, pleura_y[xs_in], s=3, c='orange', alpha=0.7,
                    label='Inliers RANSAC')

    xs_sm = np.where(~np.isnan(pleura_y_smooth))[0]
    ax_orig.plot(xs_sm, pleura_y_smooth[xs_sm], 'lime', linewidth=2,
                 label='Pleura (final)')

    ax_orig.axhline(y=y_min, color='cyan', linestyle='--', alpha=0.5,
                    label=f'Zona cautare [{y_min},{y_max}]')
    ax_orig.axhline(y=y_max, color='cyan', linestyle='--', alpha=0.5)

    for dc in debug_columns:
        ax_orig.axvline(x=dc, color='yellow', linestyle=':', alpha=0.6)

    ax_orig.set_title(
        f'Detectie pleura — confidence={confidence:.2f} — '
        f'inlieri={inlier_mask.sum()}/{W}'
    )
    ax_orig.legend(loc='upper right', fontsize=8)
    ax_orig.set_xlabel('x (px)')
    ax_orig.set_ylabel('y (px)')

    # Rand 2: imagine enhanced
    ax_enh = fig.add_subplot(gs[1, :])
    ax_enh.imshow(enhanced, cmap='gray')
    ax_enh.plot(xs_sm, pleura_y_smooth[xs_sm], 'lime', linewidth=1.5, alpha=0.8)
    for dc in debug_columns:
        ax_enh.axvline(x=dc, color='yellow', linestyle=':', alpha=0.6)
    ax_enh.set_title('Imagine dupa CLAHE')
    ax_enh.set_xlabel('x (px)')

    # Rand 3: profile pe coloane
    for i, dc in enumerate(debug_columns):
        ax = fig.add_subplot(gs[2, i])
        column = enhanced[:, dc]
        y_coords = np.arange(H)

        ax.plot(column, y_coords, 'k-', linewidth=0.8)
        ax.fill_betweenx(y_coords, 0, column, alpha=0.15, color='gray')
        ax.axhspan(y_min, y_max, color='cyan', alpha=0.1)

        if not np.isnan(pleura_y[dc]):
            y_pk = int(pleura_y[dc])
            color = 'orange' if inlier_mask[dc] else 'red'
            ax.plot(column[y_pk], y_pk, 'o', color=color, markersize=8,
                    markeredgecolor='black')
            y_sh_end = min(H, y_pk + shadow_offset_px + shadow_window_px)
            ax.axhspan(y_pk + shadow_offset_px, y_sh_end,
                       color='magenta', alpha=0.2)

        ax.invert_yaxis()
        title = (f'x={dc}\nscore={scores[dc]:.3f}'
                 if scores[dc] > -np.inf else f'x={dc}\n(nedetectat)')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('intensitate')
        if i == 0:
            ax.set_ylabel('y (px)')
        ax.set_xlim(0, 1)

    plt.suptitle(
        f'ExtractPleuralLine — Debug (one_pixel={one_pixel_mm:.3f} mm/px)',
        fontsize=13, y=0.995
    )
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════

def plot_on_image(result, img_denoised, one_pixel, contour_result=None, save_path=None):
    """
    Afiseaza imaginea CU linia pleurala si conturul suprapuse pentru verificare LOCALA.

    Daca contour_result e dat (de la ExtractPleuralContour), afiseaza si:
      - margini sus/jos
      - intreruperile evidentiate in rosu

    Daca save_path e dat, salveaza figura pe disc in loc sa o afiseze.
    """
    pleura_y = result['pleura_y']
    pleura_smooth = result['pleura_y_smooth']
    inlier_mask = result['inlier_mask']
    enhanced = result['enhanced']
    H, W = img_denoised.shape

    n_rows = 3 if contour_result is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))
    if n_rows == 2:
        axes = list(axes)

    xs = np.arange(W)
    y_min_px = _mm_to_px(DEPTH_MIN_MM, one_pixel)
    y_max_px = _mm_to_px(DEPTH_MAX_MM, one_pixel)

    # Rand 1: imaginea denoised + linia finala
    ax = axes[0]
    ax.imshow(img_denoised, cmap='gray')
    valid_raw = ~np.isnan(pleura_y)
    ax.scatter(xs[valid_raw], pleura_y[valid_raw], s=2, c='red', alpha=0.4,
               label=f'Detectii brute ({valid_raw.sum()})')
    ax.scatter(xs[inlier_mask], pleura_y[inlier_mask], s=4, c='orange', alpha=0.7,
               label=f'Inlieri RANSAC ({inlier_mask.sum()})')
    valid_smooth = ~np.isnan(pleura_smooth)
    ax.plot(xs[valid_smooth], pleura_smooth[valid_smooth], 'lime', linewidth=2,
            label=f'Pleura (smooth, {valid_smooth.sum()} col)')
    ax.axhline(y=y_min_px, color='cyan', linestyle='--', alpha=0.5,
               label=f'Zona cautare [{y_min_px}-{y_max_px} px]')
    ax.axhline(y=y_max_px, color='cyan', linestyle='--', alpha=0.5)
    ax.set_title(f'Pleura detectata (linie mediana) — '
                 f'confidence={result["confidence"]:.2f}')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')

    # Rand 2: imaginea dupa CLAHE + linia
    ax = axes[1]
    ax.imshow(enhanced, cmap='gray')
    ax.plot(xs[valid_smooth], pleura_smooth[valid_smooth], 'lime', linewidth=2)
    ax.scatter(xs[inlier_mask], pleura_y[inlier_mask], s=4, c='orange', alpha=0.6)
    ax.axhline(y=y_min_px, color='cyan', linestyle='--', alpha=0.4)
    ax.axhline(y=y_max_px, color='cyan', linestyle='--', alpha=0.4)
    ax.set_title('Imagine dupa CLAHE')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')

    # Rand 3: CONTURUL pleurei (daca e dat)
    if contour_result is not None:
        ax = axes[2]
        ax.imshow(img_denoised, cmap='gray')

        # Contururile per segment, fiecare cu culoare diferita (doar linii, FARA umplere)
        colors_seg = plt.cm.spring(np.linspace(0, 1, max(1, len(contour_result['contours_cv']))))
        for i, cnt in enumerate(contour_result['contours_cv']):
            pts = cnt.reshape(-1, 2)
            # Doar linia conturului — fara fill
            ax.plot(pts[:, 0], pts[:, 1], color=colors_seg[i], linewidth=1.5,
                    label=f'Segment {i+1} ({len(pts)//2} col)')

        # Evidentiaza intreruperile in rosu (doar linii verticale pe margini, nu fill)
        for i, (xs_i, xe_i, wmm) in enumerate(contour_result['interruptions']):
            ax.axvline(x=xs_i, color='red', linestyle='--', linewidth=1, alpha=0.7,
                       label=f'Intrerupere {i+1} ({wmm:.1f} mm)' if i < 3 else None)
            ax.axvline(x=xe_i, color='red', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_title(f'Contur pleura — {len(contour_result["segments"])} segmente, '
                     f'{len(contour_result["interruptions"])} intreruperi')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[plot_on_image] Salvat in: {save_path}")
        plt.close()
    else:
        print("\n[plot_on_image] Inchide AMBELE ferestre ca sa termine programul.")
        plt.show()


def plot_contour_zoom(contour_result, img_denoised, one_pixel, pad_mm=5.0, save_path=None):
    """
    Afiseaza conturul pleurei intr-o fereastra separata, zoom-at pe zona de interes.

    Zoom-ul e automat: include tot conturul + un buffer de pad_mm deasupra/dedesubt.
    Conturul apare ingrosat si clar, fara sa fie strivit de restul imaginii.

    Afiseaza 2 subplot-uri:
      - sus: imaginea denoised cu conturul supraimpus (zoom)
      - jos: imaginea CLAHE cu conturul supraimpus (zoom)
    """
    if not contour_result['contours_cv']:
        print("[plot_contour_zoom] Niciun contur de afisat.")
        return

    H, W = img_denoised.shape
    enhanced = (img_denoised.astype(np.float32) / 255.0) if img_denoised.dtype == np.uint8 else img_denoised

    # Calculeaza bounding box al tuturor contururilor
    all_pts = np.concatenate([c.reshape(-1, 2) for c in contour_result['contours_cv']])
    y_min = int(all_pts[:, 1].min())
    y_max = int(all_pts[:, 1].max())
    x_min = int(all_pts[:, 0].min())
    x_max = int(all_pts[:, 0].max())

    # Buffer vertical pentru context (in pixeli)
    pad_px = _mm_to_px(pad_mm, one_pixel)
    y_min = max(0, y_min - pad_px)
    y_max = min(H - 1, y_max + pad_px)

    # Buffer orizontal minim (30 px)
    x_min = max(0, x_min - 30)
    x_max = min(W - 1, x_max + 30)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    colors_seg = plt.cm.spring(np.linspace(0, 1, max(1, len(contour_result['contours_cv']))))

    for ax, img, title in [
        (axes[0], img_denoised, 'Imagine denoised — zoom pe pleura'),
        (axes[1], enhanced, 'Imagine CLAHE — zoom pe pleura'),
    ]:
        ax.imshow(img, cmap='gray', aspect='auto')

        # Contururile per segment
        for i, cnt in enumerate(contour_result['contours_cv']):
            pts = cnt.reshape(-1, 2)
            n = len(pts) // 2
            top_pts = pts[:n]
            bot_pts = pts[n:][::-1]  # reverseaza ca sa fie stanga→dreapta

            ax.plot(top_pts[:, 0], top_pts[:, 1], color=colors_seg[i],
                    linewidth=2, label=f'Segment {i+1} (sus)')
            ax.plot(bot_pts[:, 0], bot_pts[:, 1], color=colors_seg[i],
                    linewidth=2, linestyle='--', label=f'Segment {i+1} (jos)')

        # Linii verticale pentru intreruperi
        for i, (xs_i, xe_i, wmm) in enumerate(contour_result['interruptions']):
            if xs_i < x_max and xe_i > x_min:  # doar daca e in zoom
                ax.axvline(x=xs_i, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(x=xe_i, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                # Eticheta cu latimea intreruperii
                x_mid = (xs_i + xe_i) / 2
                if x_min < x_mid < x_max:
                    ax.text(x_mid, y_min + 3, f'{wmm:.1f}mm',
                            color='red', fontsize=9, ha='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # inversat (y creste in jos)
        ax.set_title(title)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        if ax is axes[0]:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    # Titlu general cu statistici
    thickness = contour_result['thickness_mm']
    valid_th = ~np.isnan(thickness)
    if valid_th.sum() > 0:
        stat_str = (f'Grosime: mean={thickness[valid_th].mean():.2f}mm, '
                    f'min={thickness[valid_th].min():.2f}mm, '
                    f'max={thickness[valid_th].max():.2f}mm  |  '
                    f'{len(contour_result["segments"])} segmente, '
                    f'{len(contour_result["interruptions"])} intreruperi')
    else:
        stat_str = 'Nicio grosime valida'

    plt.suptitle(f'Contur pleura (zoom)  —  {stat_str}', fontsize=11, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[plot_contour_zoom] Salvat in: {save_path}")
        plt.close()
    else:
        plt.show(block=False)


def main():
    orig_Image = io.imread('./ORIGINAL_IMAGES/56.jpg')

    one_pixel = PixelConverter(orig_Image)
    print(f"Un pixel = {one_pixel:.4f} mm")

    if one_pixel <= 0 or one_pixel > 0.5:
        print("\n[main] PixelConverter a dat o valoare suspecta. Folosesc fallback.")
        one_pixel = 0.07
        print(f"[main] Folosesc one_pixel = {one_pixel} mm/px (hardcodat)\n")

    crop_image = CropBorder(orig_Image)
    img_denoised = remove_noise(crop_image)

    # Pasul 1: detecteaza linia mediana a pleurei
    result = ExtractPleuralLine(img_denoised, one_pixel, debug=False)

    # Pasul 2: construieste conturul (margini sus/jos + intreruperi)
    contour_result = ExtractPleuralContour(result, img_denoised, one_pixel)

    # Raport numeric + grafice abstracte (fara imagine) — pt partajat
    print_debug_report(result, img_denoised, one_pixel)

    # FEREASTRA A: plot zoom pe contur (pentru vizualizare detaliata — NDA local)
    plot_contour_zoom(contour_result, img_denoised, one_pixel)

    # FEREASTRA B: plot full cu linia + contur (NDA local)
    plot_on_image(result, img_denoised, one_pixel, contour_result=contour_result)


def print_debug_report(result, img_denoised, one_pixel):
    """
    Afiseaza raport complet de debug FARA a afisa imaginea (pt date sub NDA).
    Include: statistici numerice + grafice abstracte (histograme, distributii).
    """
    pleura_y = result['pleura_y']
    pleura_smooth = result['pleura_y_smooth']
    scores = result['scores']
    inlier_mask = result['inlier_mask']
    enhanced = result['enhanced']
    H, W = img_denoised.shape

    # ═══════════════════════════════════════════════════════
    #  RAPORT NUMERIC
    # ═══════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  RAPORT DEBUG — ExtractPleuralLine (date anonime)")
    print("═" * 60)

    print(f"\n[Metoda]")
    print(f"  {result.get('method', 'unknown').upper()}")

    print(f"\n[Dimensiuni]")
    print(f"  Imagine: {W} x {H} px")
    print(f"  Rezolutie: {one_pixel:.4f} mm/px")
    print(f"  Zona cautare: {_mm_to_px(DEPTH_MIN_MM, one_pixel)}-{_mm_to_px(DEPTH_MAX_MM, one_pixel)} px "
          f"({DEPTH_MIN_MM}-{DEPTH_MAX_MM} mm)")

    # Statistici detectie bruta
    valid_raw = ~np.isnan(pleura_y)
    n_valid_raw = valid_raw.sum()
    print(f"\n[Detectie bruta (inainte de RANSAC)]")
    print(f"  Coloane cu varf candidat: {n_valid_raw} / {W} ({100*n_valid_raw/W:.1f}%)")
    if n_valid_raw > 0:
        print(f"  y detectat — min: {np.nanmin(pleura_y):.0f} px, "
              f"max: {np.nanmax(pleura_y):.0f} px, "
              f"median: {np.nanmedian(pleura_y):.0f} px")
        print(f"  y detectat — std: {np.nanstd(pleura_y):.1f} px "
              f"(imprastiere; <30 e bine, >60 inseamna zgomot)")

    # Statistici inlieri RANSAC
    n_inliers = inlier_mask.sum()
    print(f"\n[Dupa RANSAC]")
    print(f"  Inlieri: {n_inliers} / {W} ({100*n_inliers/W:.1f}%)")
    if n_inliers > 0:
        ys_in = pleura_y[inlier_mask]
        print(f"  y inlieri — median: {np.median(ys_in):.0f} px, "
              f"std: {np.std(ys_in):.1f} px")
        # Grosime banda de inlieri (ideal e subtire)
        iqr = np.percentile(ys_in, 75) - np.percentile(ys_in, 25)
        print(f"  y inlieri — IQR: {iqr:.0f} px (<15 e bine, sugereaza linie coerenta)")

    # Statistici finale smooth
    valid_smooth = ~np.isnan(pleura_smooth)
    n_smooth = valid_smooth.sum()
    print(f"\n[Linia finala (dupa spline)]")
    print(f"  Coloane acoperite: {n_smooth} / {W} ({100*n_smooth/W:.1f}%)")
    if n_smooth > 0:
        print(f"  y smooth — min: {np.nanmin(pleura_smooth):.0f} px, "
              f"max: {np.nanmax(pleura_smooth):.0f} px")
        print(f"  Amplitudine verticala: {np.nanmax(pleura_smooth) - np.nanmin(pleura_smooth):.0f} px")

    # Statistici scoruri
    valid_scores = scores[scores > -np.inf]
    if len(valid_scores) > 0:
        print(f"\n[Scoruri umbra acustica]")
        print(f"  Media (toti): {valid_scores.mean():.3f}")
        print(f"  Media (inlieri): {scores[inlier_mask].mean():.3f}" if n_inliers > 0 else "  (niciun inlier)")
        print(f"  Min / Max: {valid_scores.min():.3f} / {valid_scores.max():.3f}")
        print(f"  Interpretare: >0.2 = umbra clara, 0.1-0.2 = moderata, <0.1 = slaba")

    print(f"\n[Scor final]")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Contur OpenCV: {result['contour'].shape}")

    # Diagnostic automat
    print(f"\n[Diagnostic automat]")
    if n_inliers < W * 0.3:
        print(f"  ⚠ Prea putini inlieri ({100*n_inliers/W:.0f}%) — poate ajusta DEPTH_MIN/MAX_MM")
    if n_valid_raw > 0 and np.nanstd(pleura_y) > 60:
        print(f"  ⚠ Imprastiere mare a detectiilor brute — zgomot sau fascie puternica")
    if valid_scores.mean() < 0.1 if len(valid_scores) > 0 else False:
        print(f"  ⚠ Scoruri umbra mici — poate mari SHADOW_WINDOW_MM sau scade PEAK_PROMINENCE")
    if n_inliers >= W * 0.5 and result['confidence'] > 0.5:
        print(f"  ✓ Detectie buna")

    print("═" * 60 + "\n")

    # ═══════════════════════════════════════════════════════
    #  GRAFICE ABSTRACTE (NU CONTIN IMAGINEA)
    # ═══════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Histograma pozitiilor y detectate (bruta vs inlieri)
    ax = axes[0, 0]
    if valid_raw.sum() > 0:
        ax.hist(pleura_y[valid_raw], bins=50, alpha=0.5,
                label=f'Brute (n={valid_raw.sum()})', color='red')
    if inlier_mask.sum() > 0:
        ax.hist(pleura_y[inlier_mask], bins=50, alpha=0.7,
                label=f'Inlieri RANSAC (n={inlier_mask.sum()})', color='orange')
    ax.axvline(_mm_to_px(DEPTH_MIN_MM, one_pixel), color='cyan',
               linestyle='--', label='Limite cautare')
    ax.axvline(_mm_to_px(DEPTH_MAX_MM, one_pixel), color='cyan', linestyle='--')
    ax.set_xlabel('y (px) — adancime in imagine')
    ax.set_ylabel('Numar coloane')
    ax.set_title('Distributia pozitiilor y detectate')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Histograma scorurilor de umbra acustica
    ax = axes[0, 1]
    if len(valid_scores) > 0:
        ax.hist(valid_scores, bins=40, alpha=0.5,
                label='Toate detectiile', color='blue')
    if inlier_mask.sum() > 0:
        ax.hist(scores[inlier_mask], bins=40, alpha=0.7,
                label='Doar inlieri', color='green')
    ax.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='Prag "slab"')
    ax.axvline(0.2, color='red', linestyle='--', alpha=0.5, label='Prag "clar"')
    ax.set_xlabel('Scor umbra acustica (intensitate_varf - mu_sub)')
    ax.set_ylabel('Numar coloane')
    ax.set_title('Distributia scorurilor de umbra acustica')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. y detectat vs coloana (profil "long axis")
    ax = axes[1, 0]
    xs = np.arange(W)
    xs_raw = xs[valid_raw]
    ax.scatter(xs_raw, pleura_y[valid_raw], s=3, alpha=0.3, color='red',
               label=f'Brute ({valid_raw.sum()})')
    xs_in = xs[inlier_mask]
    ax.scatter(xs_in, pleura_y[inlier_mask], s=4, alpha=0.6, color='orange',
               label=f'Inlieri ({inlier_mask.sum()})')
    xs_sm = xs[valid_smooth]
    ax.plot(xs_sm, pleura_smooth[valid_smooth], 'lime', linewidth=2,
            label='Smooth final')
    ax.axhline(_mm_to_px(DEPTH_MIN_MM, one_pixel), color='cyan',
               linestyle='--', alpha=0.5)
    ax.axhline(_mm_to_px(DEPTH_MAX_MM, one_pixel), color='cyan',
               linestyle='--', alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel('x (px) — coloana')
    ax.set_ylabel('y (px) — adancime detectata')
    ax.set_title('Pozitia pleurei pe latimea imaginii')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Profile intensitate 5 coloane (doar curbele, FARA imagine)
    ax = axes[1, 1]
    debug_cols = np.linspace(int(W * 0.15), int(W * 0.85), 5, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(debug_cols)))
    for col_x, c in zip(debug_cols, colors):
        profile = enhanced[:, col_x]
        ax.plot(profile, np.arange(H), color=c, linewidth=0.8,
                label=f'x={col_x}')
        # Marker pentru pozitia detectata
        if not np.isnan(pleura_y[col_x]):
            y_pk = int(pleura_y[col_x])
            marker = 'o' if inlier_mask[col_x] else 'x'
            ax.plot(profile[y_pk], y_pk, marker, color=c, markersize=10,
                    markeredgecolor='black', markeredgewidth=1)
    ax.axhspan(_mm_to_px(DEPTH_MIN_MM, one_pixel),
               _mm_to_px(DEPTH_MAX_MM, one_pixel),
               color='cyan', alpha=0.1, label='Zona cautare')
    ax.invert_yaxis()
    ax.set_xlabel('Intensitate (0-1)')
    ax.set_ylabel('y (px)')
    ax.set_title('Profile de intensitate pe 5 coloane (o=inlier, x=outlier)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle(f'ExtractPleuralLine — Debug anonim (nu contine imaginea)',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.show(block=False)  # NU blocheaza — permite deschiderea a doua figuri


if __name__ == '__main__':
    main()