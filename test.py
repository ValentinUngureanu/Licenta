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

# RANSAC
RANSAC_POLY_DEGREE = 2
RANSAC_THRESH_PX = 10     # mai strict — reduce imprastierea inlierilor
RANSAC_ITERATIONS = 300   # mai multe iteratii pt stabilitate
RANSAC_MIN_INLIERS_FRAC = 0.3


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


# ═════════════════════════════════════════════════════════════
#  4. EXTRACT PLEURAL LINE — functia principala
# ═════════════════════════════════════════════════════════════

def ExtractPleuralLine(crop_image, one_pixel_mm, debug=False, debug_columns=None):
    """
    Detecteaza linia pleurala prin scanline A-mode + criteriu de umbra acustica.

    Returneaza dict cu:
        pleura_y, pleura_y_smooth, contour, scores, inlier_mask,
        confidence, enhanced
    """
    # ─ 1. Preprocessing
    gray = _to_gray_float(crop_image)
    enhanced = _apply_clahe(gray)
    H, W = enhanced.shape

    # Safety: one_pixel_mm invalid (0 sau negativ)
    if one_pixel_mm <= 0:
        raise ValueError(
            f"one_pixel_mm invalid: {one_pixel_mm}. "
            f"PixelConverter a esuat — verifica OCR-ul si detectia tick-urilor, "
            f"sau hardcodeaza o valoare (tipic 0.05-0.10 mm/px pt ecografie)."
        )

    y_min = max(0, _mm_to_px(DEPTH_MIN_MM, one_pixel_mm))
    y_max = min(H - 1, _mm_to_px(DEPTH_MAX_MM, one_pixel_mm))
    if y_max <= y_min + 10:
        raise ValueError(
            f"Zona cautare prea mica: y_min={y_min}, y_max={y_max}, H={H}. "
            f"Verifica one_pixel_mm={one_pixel_mm}."
        )

    shadow_offset_px = max(2, _mm_to_px(SHADOW_OFFSET_MM, one_pixel_mm))
    shadow_window_px = max(5, _mm_to_px(SHADOW_WINDOW_MM, one_pixel_mm))

    # ─ 2. Scoreaza fiecare coloana
    pleura_y = np.full(W, np.nan, dtype=np.float64)
    scores = np.full(W, -np.inf, dtype=np.float64)

    for x in range(W):
        y_best, score_best = _score_column(
            enhanced[:, x], y_min, y_max, shadow_offset_px, shadow_window_px
        )
        if y_best is not None:
            pleura_y[x] = y_best
            scores[x] = score_best

    # ─ 3. RANSAC
    valid_mask = ~np.isnan(pleura_y)
    xs_valid = np.where(valid_mask)[0]
    ys_valid = pleura_y[valid_mask]

    if len(xs_valid) < W * RANSAC_MIN_INLIERS_FRAC:
        print(f"[ExtractPleuralLine] Putine detectii: {len(xs_valid)}/{W}")

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
        print("[ExtractPleuralLine] RANSAC esuat, folosesc toate detectiile.")
        inlier_mask = valid_mask.copy()

    # ─ 4. Smoothing + interpolare
    xs_inlier = np.where(inlier_mask)[0]
    ys_inlier = pleura_y[inlier_mask]
    pleura_y_smooth = np.full(W, np.nan)

    if len(xs_inlier) >= 4:
        ys_med = medfilt(ys_inlier, kernel_size=5) if len(ys_inlier) >= 5 else ys_inlier
        try:
            spline = UnivariateSpline(xs_inlier, ys_med, k=3, s=len(xs_inlier))
            # Acopera DOAR coloanele cu inlieri reali in apropiere (max 30 px gap)
            # Evita extrapolarea in zonele fara suport (ex: capetele imaginii)
            x_min, x_max = xs_inlier.min(), xs_inlier.max()
            x_full = np.arange(x_min, x_max + 1)
            y_full = spline(x_full)

            # Constrange y_full sa nu iasa din range-ul inlierilor (+-10 px toleranta)
            y_lo = ys_inlier.min() - 10
            y_hi = ys_inlier.max() + 10
            mask_reasonable = (y_full >= y_lo) & (y_full <= y_hi)

            valid_x = x_full[mask_reasonable]
            valid_y = y_full[mask_reasonable]
            pleura_y_smooth[valid_x] = valid_y

            n_rejected = (~mask_reasonable).sum()
            if n_rejected > 0:
                print(f"[ExtractPleuralLine] Spline respins pt {n_rejected} coloane (extrapolare prea mare)")
        except Exception as e:
            print(f"[ExtractPleuralLine] Spline esuat ({e}), folosesc interpolare liniara.")
            x_full = np.arange(W)
            pleura_y_smooth = np.interp(x_full, xs_inlier, ys_inlier,
                                         left=np.nan, right=np.nan)
    else:
        print(f"[ExtractPleuralLine] Prea putini inlieri ({len(xs_inlier)}).")

    # ─ 5. Contur format OpenCV (pt Width_and_Irreg)
    valid_smooth = ~np.isnan(pleura_y_smooth)
    xs_c = np.where(valid_smooth)[0]
    ys_c = pleura_y_smooth[valid_smooth].astype(np.int32)
    contour = np.stack([xs_c, ys_c], axis=-1).reshape(-1, 1, 2)

    # ─ 6. Confidence
    if inlier_mask.sum() > 0:
        confidence = float(np.clip(scores[inlier_mask].mean() / 0.3, 0.0, 1.0))
    else:
        confidence = 0.0

    # ─ 7. Debug plot
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

def plot_on_image(result, img_denoised, one_pixel, save_path=None):
    """
    Afiseaza imaginea CU linia pleurala suprapusa pentru verificare vizuala LOCALA.

    ATENTIE: afiseaza imaginea originala — foloseste doar pentru debug local,
    nu trimite screenshot-urile nimanui daca sunt sub NDA.

    Daca save_path e dat, salveaza figura pe disc in loc sa o afiseze.
    """
    pleura_y = result['pleura_y']
    pleura_smooth = result['pleura_y_smooth']
    inlier_mask = result['inlier_mask']
    enhanced = result['enhanced']
    H, W = img_denoised.shape

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Rand 1: imaginea denoised + linia finala
    ax = axes[0]
    ax.imshow(img_denoised, cmap='gray')

    xs = np.arange(W)

    # Detectii brute (rosu, mic)
    valid_raw = ~np.isnan(pleura_y)
    ax.scatter(xs[valid_raw], pleura_y[valid_raw], s=2, c='red', alpha=0.4,
               label=f'Detectii brute ({valid_raw.sum()})')

    # Inlieri RANSAC (portocaliu)
    ax.scatter(xs[inlier_mask], pleura_y[inlier_mask], s=4, c='orange', alpha=0.7,
               label=f'Inlieri RANSAC ({inlier_mask.sum()})')

    # Linia finala (verde)
    valid_smooth = ~np.isnan(pleura_smooth)
    ax.plot(xs[valid_smooth], pleura_smooth[valid_smooth], 'lime', linewidth=2,
            label=f'Pleura (smooth, {valid_smooth.sum()} col)')

    # Zona de cautare
    y_min_px = _mm_to_px(DEPTH_MIN_MM, one_pixel)
    y_max_px = _mm_to_px(DEPTH_MAX_MM, one_pixel)
    ax.axhline(y=y_min_px, color='cyan', linestyle='--', alpha=0.5,
               label=f'Zona cautare [{y_min_px}-{y_max_px} px]')
    ax.axhline(y=y_max_px, color='cyan', linestyle='--', alpha=0.5)

    ax.set_title(f'Pleura detectata peste imaginea denoised — '
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
    ax.set_title('Imagine dupa CLAHE (aceeasi linie)')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[plot_on_image] Salvat in: {save_path}")
        plt.close()
    else:
        # block=True (default) — programul asteapta aici pana inchizi
        # ambele ferestre. Prima fereastra e deja deschisa (block=False).
        print("\n[plot_on_image] Inchide AMBELE ferestre ca sa termine programul.")
        plt.show()


def main():
    orig_Image = io.imread('./ORIGINAL_IMAGES/1.jpg')

    one_pixel = PixelConverter(orig_Image)
    print(f"Un pixel = {one_pixel:.4f} mm")

    # Safety: daca OCR-ul a esuat, foloseste o valoare hardcodata
    if one_pixel <= 0 or one_pixel > 0.5:
        print("\n[main] PixelConverter a dat o valoare suspecta. Folosesc fallback.")
        one_pixel = 0.07
        print(f"[main] Folosesc one_pixel = {one_pixel} mm/px (hardcodat)\n")

    crop_image = CropBorder(orig_Image)
    img_denoised = remove_noise(crop_image)

    result = ExtractPleuralLine(img_denoised, one_pixel, debug=False)

    # Raport numeric + grafice abstracte (fara imagine) — pt partajat
    print_debug_report(result, img_denoised, one_pixel)

    # Plot pe imagine — DOAR pentru verificare locala (NDA-safe)
    plot_on_image(result, img_denoised, one_pixel)


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