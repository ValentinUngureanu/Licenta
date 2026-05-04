import os
import re
import math
import traceback

import cv2
import numpy as np
import pytesseract as tes

from skimage import io
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_yen

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = r"C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES"
OUTPUT_ROOT = r"C:\Facultate\AN4\Licenta\Licenta-Cod\DEBUG_OUT_AUTO_MODE"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

START_IDX = 0
END_IDX = 61

ONE_PIXEL_FALLBACK = 0.07


# ============================================================
# MODURI AUTOMATE
# ============================================================
# Nu mai folosim GOOD_IDS / SURPLUS_IDS / PATHOLOGIC_IDS / WRONG_IDS.
# Pentru fiecare imagine rulam toate modurile si alegem automat rezultatul
# cu scorul cel mai bun.

AUTO_MODES = ["normal", "surplus", "pathologic"]


# ============================================================
# BASIC HELPERS
# ============================================================

class Point:
    x = 0.0   # row
    y = 0.0   # col
    dist = 0.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_rgb(path, img_rgb):
    if img_rgb.ndim == 2:
        cv2.imwrite(path, img_rgb)
    else:
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def to_gray_uint8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    return img


# ============================================================
# CROP AND PIXEL CONVERTER
# ============================================================

def CropBorder(orig_Image):
    orig_img = orig_Image.copy()

    if orig_img.ndim == 2:
        gray_Image = orig_img.copy()
    else:
        gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

    _, img_bin = cv2.threshold(gray_Image, 128, 255, cv2.THRESH_BINARY)
    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:
            approx = cv2.approxPolyDP(
                cnt,
                0.01 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    black = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:
            approx = cv2.approxPolyDP(
                cnt,
                0.01 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(black, [approx], 0, 255, -1)

            if len(approx) == 2:
                cv2.drawContours(black, [approx], 0, 255, -1)

    white_pixels = np.array(np.where(black == 255))

    if white_pixels.shape[1] == 0:
        last_small = np.array([])
    else:
        last_small = white_pixels[1, white_pixels[1] < 100]

    if len(white_pixels[1]) == 0 or len(last_small) == 0:
        left_bound = 25
    else:
        left_bound = int(last_small[-1])

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 30:
            approx = cv2.approxPolyDP(
                cnt,
                0.00001 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    black = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 30:
            approx = cv2.approxPolyDP(
                cnt,
                0.00001 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(black, [approx], 0, 255, -1)

            if len(approx) == 2:
                cv2.drawContours(black, [approx], 0, 255, -1)

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

    img_temp2 = cv2.erode(black, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    bar_image = horizontal_lines_img

    columns = np.zeros(bar_image.shape[1], dtype=int)

    for i in range(bar_image.shape[1]):
        columns[i] = np.count_nonzero(bar_image[:, i])

    if np.max(columns) == 0:
        print("[CropBorder] Bara nu a fost gasita. Returnez imaginea intreaga.")
        return gray_Image.copy()

    bar_pos = np.where(columns == np.max(columns))[0][0]

    bar = bar_image[:, bar_pos] // 255
    bar_pixels = np.array(np.where(bar == 1))

    if bar_pixels.shape[1] == 0:
        print("[CropBorder] Pixelii barei nu au fost gasiti. Returnez imaginea intreaga.")
        return gray_Image.copy()

    first_bar_pixel = int(bar_pixels[:, 0][0])
    last_bar_pixel = int(bar_pixels[:, -1][0])

    if first_bar_pixel == 0 or last_bar_pixel == 0:
        print("[CropBorder] Imagine neconforma. Returnez imaginea intreaga.")
        return gray_Image.copy()

    x2 = int(bar_pos - 20)

    if x2 <= left_bound:
        print("[CropBorder] Coordonate crop invalide. Returnez imaginea intreaga.")
        return gray_Image.copy()

    crop_img = gray_Image[
        first_bar_pixel:last_bar_pixel,
        left_bound:x2
    ].copy()

    if crop_img.size == 0 or crop_img.shape[0] < 30 or crop_img.shape[1] < 30:
        print("[CropBorder] Crop gol sau prea mic. Returnez imaginea intreaga.")
        return gray_Image.copy()

    return crop_img


def PixelConverter(orig_Image):
    try:
        orig_img = orig_Image.copy()

        if orig_img.ndim == 2:
            gray_Image = orig_img.copy()
        else:
            gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

        _, img_bin = cv2.threshold(
            gray_Image,
            128,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        _, threshold = cv2.threshold(gray_Image, 110, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        black = np.zeros_like(img_bin)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 20:
                approx = cv2.approxPolyDP(
                    cnt,
                    0.00001 * cv2.arcLength(cnt, True),
                    True
                )

                if len(approx) == 4:
                    cv2.drawContours(black, [approx], 0, 255, -1)

                if len(approx) == 2:
                    cv2.drawContours(black, [approx], 0, 255, -1)

        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

        img_temp2 = cv2.erode(black, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

        columns = np.zeros(horizontal_lines_img.shape[1], dtype=int)

        for i in range(horizontal_lines_img.shape[1]):
            columns[i] = np.count_nonzero(horizontal_lines_img[:, i])

        if np.max(columns) == 0:
            return ONE_PIXEL_FALLBACK

        bar_pos = np.where(columns == np.max(columns))[0][0]
        bar = horizontal_lines_img[:, bar_pos] // 255

        indices = [i for i, x in enumerate(bar) if x == 1]

        if len(indices) < 2:
            return ONE_PIXEL_FALLBACK

        tes.pytesseract.tesseract_cmd = TESSERACT_CMD

        try:
            image_str = tes.image_to_string(np.invert(img_bin))
        except Exception:
            return ONE_PIXEL_FALLBACK

        depth = None

        if "cm" in image_str:
            depth_str = image_str[image_str.find("cm") - 5:image_str.find("cm")]
            depth_no = [int(i) for i in depth_str if i.isdigit()]

            if len(depth_no) > 0:
                depth = depth_no[0] * 10

        if depth is None and "mm" in image_str:
            depth_str = image_str[image_str.find("mm") - 5:image_str.find("mm")]
            depth_no = [int(s) for s in re.findall(r"\b\d+\b", depth_str)]

            if len(depth_no) > 0:
                depth = depth_no[0]

        if depth is None:
            if "\nD " in image_str:
                depth_str = image_str[
                    image_str.find("\nD "):image_str.find("\nD ") + 6
                ]
                depth_no = [int(i) for i in depth_str if i.isdigit()]

                if len(depth_no) > 0:
                    if depth_no[0] < 50:
                        depth = depth_no[0] * 10
                    else:
                        depth = depth_no[0]

            if depth is None and "-D " in image_str:
                depth_str = image_str[
                    image_str.find("-D "):image_str.find("-D ") + 6
                ]
                depth_no = [int(i) for i in depth_str if i.isdigit()]

                if len(depth_no) > 0:
                    if depth_no[0] < 50:
                        depth = depth_no[0] * 10
                    else:
                        depth = depth_no[0]

        if depth is None:
            return ONE_PIXEL_FALLBACK

        tick_sum = np.sum(np.diff(indices))

        if tick_sum <= 0:
            return ONE_PIXEL_FALLBACK

        depth_pix = depth / tick_sum

        if depth_pix <= 0 or depth_pix > 0.5:
            return ONE_PIXEL_FALLBACK

        return depth_pix

    except Exception:
        return ONE_PIXEL_FALLBACK


# ============================================================
# IMAGE PROCESSING CORE
# ============================================================

def ReduceColorPalette(image, nr_of_colors):
    image = image.copy()

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    pixels = image.reshape(-1)

    if len(pixels) == 0:
        return image

    pixels_sorted = np.sort(pixels)[::-1]
    nr_pixel_per_color = max(1, int(len(pixels_sorted) / nr_of_colors))

    colors = []

    for i in range(nr_of_colors + 1):
        idx = i * nr_pixel_per_color

        if idx >= len(pixels_sorted):
            color = pixels_sorted[-1]
        else:
            color = pixels_sorted[idx]

        colors.append(color)

    flat = image.reshape(-1)
    order = np.argsort(flat)[::-1]
    out = flat.copy()

    for p_idx, flat_idx in enumerate(order):
        col = int(p_idx / nr_pixel_per_color)

        if col >= len(colors):
            col = len(colors) - 1

        out[flat_idx] = colors[col]

    return out.reshape(image.shape)


def Binarize(Image, tresh):
    if Image.ndim == 3:
        Image = rgb2gray(Image)
    else:
        Image = Image.astype(np.float32)

        if np.max(Image) > 1:
            Image = Image / 255.0

    if tresh > 1:
        tresh = tresh / 255.0

    binarized = Image < tresh

    return binarized


def ComputeAdaptivePleuraThreshold(img, mode="normal"):
    """
    Prag mai stabil decat np.max(img) - 10.

    In codul vechi, daca exista un singur pixel foarte luminos, pragul devenea prea sus
    si se pierdeau fragmente din pleura. Aici combinam Yen cu percentile.

    - surplus: prag mai strict, ca sa reduca marcajele in plus
    - pathologic: prag mai permisiv, ca sa pastreze bucati intrerupte/neregulate
    - normal: balans intre cele doua
    """
    img_u = to_gray_uint8(img)

    if img_u.size == 0:
        return 245.0

    values = img_u.reshape(-1).astype(np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 245.0

    try:
        yen_thr = float(threshold_yen(img_u))
    except Exception:
        yen_thr = float(np.percentile(values, 92))

    if mode == "surplus":
        percentile_thr = float(np.percentile(values, 95.0))
        offset = 2.0
    elif mode == "pathologic":
        percentile_thr = float(np.percentile(values, 88.0))
        offset = -3.0
    else:
        percentile_thr = float(np.percentile(values, 92.0))
        offset = 0.0

    low_guard = float(np.percentile(values, 72.0))
    high_guard = float(np.percentile(values, 99.2))

    threshold = 0.55 * yen_thr + 0.45 * percentile_thr + offset
    threshold = max(low_guard, min(high_guard, threshold))

    # Evitam praguri extreme.
    threshold = max(5.0, min(250.0, threshold))

    return threshold


def CleanFinalPleuraMask(mask, image_shape, mode="normal"):
    """
    Curatare conservatoare a mastii finale.

    Scop:
    - elimina fragmente minuscule sau evident verticale;
    - in modul surplus pastreaza mai strict componentele lungi/subtiri;
    - in modul pathologic permite mai multe componente separate.
    """
    if mask is None or mask.size == 0:
        return mask

    h, w = image_shape[:2]
    mask_bin = (mask > 0).astype(np.uint8) * 255

    if np.count_nonzero(mask_bin) < 5:
        return mask_bin

    if mode == "surplus":
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, open_kernel)
    elif mode == "pathologic":
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_kernel)
    else:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask_bin > 0).astype(np.uint8),
        connectivity=8
    )

    if n_labels <= 1:
        return mask_bin

    candidates = []

    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < 8:
            continue

        width_frac = bw / max(w, 1)
        height_frac = bh / max(h, 1)
        area_frac = area / max(h * w, 1)
        aspect = bw / max(bh, 1)
        y_center = (y + bh / 2.0) / max(h, 1)

        if mode == "surplus":
            min_width_frac = 0.020
            max_height_frac = 0.22
            max_area_frac = 0.12
            min_aspect = 1.35
            max_keep = 3
        elif mode == "pathologic":
            min_width_frac = 0.006
            max_height_frac = 0.55
            max_area_frac = 0.25
            min_aspect = 0.45
            max_keep = 10
        else:
            min_width_frac = 0.012
            max_height_frac = 0.38
            max_area_frac = 0.18
            min_aspect = 0.75
            max_keep = 5

        if width_frac < min_width_frac:
            continue

        if height_frac > max_height_frac:
            continue

        if area_frac > max_area_frac:
            continue

        if aspect < min_aspect:
            continue

        # Evitam componente foarte aproape de margini, dar nu le eliminam agresiv.
        y_score = 1.0
        if y_center < 0.05 or y_center > 0.95:
            y_score = 0.35

        score = (
            2.6 * min(1.0, width_frac / 0.55)
            + 1.2 * min(1.0, aspect / 8.0)
            + 0.7 * y_score
            - 1.6 * height_frac
            - 2.0 * area_frac
        )

        candidates.append((score, label, max_keep))

    if len(candidates) == 0:
        # Daca filtrarea a fost prea dura, pastram masca initiala ca sa nu pierdem pleura.
        return mask_bin

    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    max_keep = candidates[0][2]
    keep_labels = [label for _, label, _ in candidates[:max_keep]]

    cleaned = np.zeros_like(mask_bin)

    for label in keep_labels:
        cleaned[labels == label] = 255

    if np.count_nonzero(cleaned) < 5:
        return mask_bin

    return cleaned


def IdnetifyPoly(X_, Y_, order):
    X_ = np.asarray(X_, dtype=np.float64)
    Y_ = np.asarray(Y_, dtype=np.float64)

    valid = np.isfinite(X_) & np.isfinite(Y_)
    X_ = X_[valid]
    Y_ = Y_[valid]

    if len(X_) == 0 or len(Y_) == 0:
        p = np.poly1d([0])
        return p, np.array([])

    if len(X_) < 2 or len(np.unique(Y_)) < 2:
        const_x = float(np.mean(X_))
        p = np.poly1d([const_x])
        return p, np.ones_like(Y_) * const_x

    safe_order = min(order, len(X_) - 1, len(np.unique(Y_)) - 1)

    try:
        coeff = np.polyfit(Y_, X_, safe_order)
        p = np.poly1d(coeff)
        fitted_X = p(Y_)
        return p, fitted_X

    except Exception:
        const_x = float(np.median(X_))
        p = np.poly1d([const_x])
        fitted_X = np.ones_like(Y_) * const_x
        return p, fitted_X


def Fit(pts, poly_line, deviation):
    point = []

    for p in pts:
        x_hat = poly_line(p.y)

        if abs(x_hat - p.x) < deviation:
            point.append(p)

    return point


def Fit2(pts, poly_line, deviation, X, Y):
    point = []
    X = []
    Y = []
    max_dev = 300

    for p in pts:
        x_hat = poly_line(p.y)

        if abs(x_hat - p.x) < deviation:
            point.append(p)
            X.append(p.x)
            Y.append(p.y)

        elif abs(x_hat - p.x) < max_dev:
            if x_hat - p.x < 0:
                p.x = x_hat + deviation
            else:
                p.x = x_hat - deviation

            point.append(p)
            X.append(p.x)
            Y.append(p.y)

    return point, X, Y


def ComputeDistance(p1, p2):
    return math.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2))


def removeOutliers(points, outlierConstant):
    if len(points) < 2:
        return np.zeros((0, 0))

    matrx = np.zeros(shape=(len(points), len(points)))

    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                dist = int(ComputeDistance(points[i], points[j]))

                if dist <= outlierConstant:
                    matrx[i][j] = 1

    return matrx


def ExtractContour(Image, points):
    if Image.ndim == 3:
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)

    for y in range(0, Image.shape[1] - 1, 1):
        for x in range(Image.shape[0] - 1, 0, -1):
            if Image[x, y] == 1:
                p = Point()
                p.x = x
                p.y = y
                points.append(p)
                break


def ExtractConnectedComponents(distances, points):
    X_ = []
    Y_ = []
    pts = []

    if len(points) < 2 or distances.size == 0:
        return pts, X_, Y_

    graph = csr_matrix(distances)

    n_components, labels = connected_components(
        csgraph=graph,
        directed=False,
        return_labels=True
    )

    max_nr = 0
    lbl = 0

    for i in range(0, n_components):
        nr_pts = np.count_nonzero(labels == i)

        if nr_pts > max_nr:
            max_nr = nr_pts
            lbl = i

    for i in range(0, len(points)):
        if labels[i] == lbl:
            X_.append(points[i].x)
            Y_.append(points[i].y)
            pts.append(points[i])

    return pts, X_, Y_


def ConnectedContour(ROI):
    if ROI is False or ROI is None:
        return False

    img = img_as_ubyte(ROI)

    if img.size == 0:
        return False

    try:
        thresh = threshold_yen(img) - 0.2 * threshold_yen(img)
    except Exception:
        return False

    binary = img > thresh
    img = binary.astype(np.uint8)
    img *= 255

    new_img = np.zeros_like(img)

    for val in np.unique(img)[1:]:
        mask = np.uint8(img == val)

        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]

        if stats.shape[0] <= 1:
            continue

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_img[labels == largest_label] = val

    img = cv2.dilate(new_img, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours is None or len(contours) == 0:
        return False

    return contours


def FindAllContoursFromMask(mask):
    if mask is None or mask.size == 0:
        return []

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours is None:
        return []

    return contours


# ============================================================
# COMPONENT FILTERS
# ============================================================

def ComponentStats(component):
    if component is False or component is None:
        return None

    pts = component.reshape(-1, 2)

    if len(pts) < 5:
        return None

    cols = pts[:, 0].astype(np.float64)
    rows = pts[:, 1].astype(np.float64)

    width = float(np.max(cols) - np.min(cols) + 1)
    height = float(np.max(rows) - np.min(rows) + 1)
    area = float(cv2.contourArea(component))

    if len(np.unique(cols)) > 1:
        try:
            slope = abs(float(np.polyfit(cols, rows, 1)[0]))
        except Exception:
            slope = 999.0
    else:
        slope = 999.0

    return {
        "cols": cols,
        "rows": rows,
        "width": width,
        "height": height,
        "area": area,
        "slope": slope,
        "center_row": float(np.median(rows)),
        "center_col": float(np.median(cols)),
        "min_x": int(np.min(cols)),
        "max_x": int(np.max(cols)),
        "min_y": int(np.min(rows)),
        "max_y": int(np.max(rows))
    }


def IsPleuraLikeComponent(component, principal_poly=None, image_shape=None, mode="normal"):
    stats = ComponentStats(component)

    if stats is None:
        return False

    width = stats["width"]
    height = stats["height"]
    area = stats["area"]
    slope = stats["slope"]

    h = 1
    w = 1

    if image_shape is not None:
        h, w = image_shape[:2]

    if mode == "surplus":
        min_width = max(18, 0.018 * w)
        max_height = max(55, 0.18 * h)
        max_ratio = 0.70
        max_slope = 0.80
        max_dev = max(65, 0.12 * h)

    elif mode == "pathologic":
        min_width = max(5, 0.004 * w)
        max_height = max(155, 0.42 * h)
        max_ratio = 1.85
        max_slope = 1.85
        max_dev = max(190, 0.30 * h)

    else:
        min_width = max(12, 0.01 * w)
        max_height = max(100, 0.28 * h)
        max_ratio = 1.10
        max_slope = 1.20
        max_dev = max(120, 0.20 * h)

    if width < min_width:
        return False

    if height > max_height:
        return False

    if height / max(width, 1) > max_ratio:
        return False

    if slope > max_slope:
        return False

    if area < 5:
        return False

    if principal_poly is not None:
        cols = stats["cols"]
        rows = stats["rows"]

        try:
            expected_rows = principal_poly(cols)
            median_dev = float(np.median(np.abs(rows - expected_rows)))

            if median_dev > max_dev:
                return False

        except Exception:
            return False

    return True


def IsPrincipalComponentPlausible(component, image_shape=None):
    stats = ComponentStats(component)

    if stats is None:
        return False

    width = stats["width"]
    height = stats["height"]
    area = stats["area"]
    slope = stats["slope"]

    if image_shape is None:
        return True

    h, w = image_shape[:2]

    if width < 0.04 * w:
        return False

    if height > 0.60 * h:
        return False

    if area > 0.50 * h * w:
        return False

    if slope > 2.20:
        return False

    return True


def FinalMaskLooksPlausible(mask, image_shape, mode="normal"):
    if mask is None or mask.size == 0:
        return False

    ys, xs = np.where(mask > 0)

    if len(xs) < 5:
        return False

    h, w = image_shape[:2]

    width = np.max(xs) - np.min(xs) + 1
    height = np.max(ys) - np.min(ys) + 1
    area = np.count_nonzero(mask)

    if width < 0.06 * w:
        return False

    if mode == "pathologic":
        if height > 0.65 * h:
            return False

        if area > 0.45 * h * w:
            return False

    else:
        if height > 0.48 * h:
            return False

        if area > 0.35 * h * w:
            return False

    return True


# ============================================================
# AUTOMATIC MODE SELECTION
# ============================================================

def ComputeMaskMetrics(mask, image_shape):
    h, w = image_shape[:2]

    metrics = {
        "has_pixels": False,
        "width_frac": 0.0,
        "height_frac": 0.0,
        "area_frac": 0.0,
        "coverage": 0.0,
        "continuity": 0.0,
        "y_center_frac": 0.0,
        "median_thickness": 0.0,
        "thickness_iqr": 0.0,
        "thickness_var_norm": 0.0,
        "components_count": 0,
        "slope": 999.0
    }

    if mask is None or mask.size == 0:
        return metrics

    mask_bin = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)

    if len(xs) < 5:
        return metrics

    metrics["has_pixels"] = True

    x_min = int(np.min(xs))
    x_max = int(np.max(xs))
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = int(np.count_nonzero(mask_bin))

    metrics["width_frac"] = width / max(w, 1)
    metrics["height_frac"] = height / max(h, 1)
    metrics["area_frac"] = area / max(h * w, 1)
    metrics["y_center_frac"] = float(np.median(ys)) / max(h, 1)

    unique_x = np.unique(xs)
    metrics["coverage"] = len(unique_x) / max(w, 1)
    metrics["continuity"] = len(unique_x) / max(width, 1)

    thicknesses = []
    centers_x = []
    centers_y = []

    for x in unique_x:
        col_ys = ys[xs == x]

        if len(col_ys) == 0:
            continue

        y1 = int(np.min(col_ys))
        y2 = int(np.max(col_ys))

        thicknesses.append(y2 - y1 + 1)
        centers_x.append(x)
        centers_y.append((y1 + y2) / 2.0)

    if len(thicknesses) > 0:
        thicknesses = np.asarray(thicknesses, dtype=np.float64)

        med = float(np.median(thicknesses))
        q25 = float(np.percentile(thicknesses, 25))
        q75 = float(np.percentile(thicknesses, 75))

        metrics["median_thickness"] = med
        metrics["thickness_iqr"] = q75 - q25
        metrics["thickness_var_norm"] = (q75 - q25) / max(med, 1.0)

    try:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_bin,
            connectivity=8
        )
        metrics["components_count"] = max(0, n_labels - 1)
    except Exception:
        metrics["components_count"] = 999

    try:
        if len(centers_x) >= 2 and len(np.unique(centers_x)) >= 2:
            coeff = np.polyfit(
                np.asarray(centers_x, dtype=np.float64),
                np.asarray(centers_y, dtype=np.float64),
                1
            )
            metrics["slope"] = abs(float(coeff[0]))
    except Exception:
        metrics["slope"] = 999.0

    return metrics


def ScorePleuraResult(result, image_shape, mode):
    mask = result["display_mask"]
    m = ComputeMaskMetrics(mask, image_shape)

    if not m["has_pixels"]:
        m["score"] = -999.0
        m["accepted"] = False
        m["suspect"] = True
        m["reason"] = "empty_mask"
        m["mode"] = mode
        m["interruptions_count"] = 0
        m["nodules_count"] = 0
        m["raw_components_count"] = 0
        return -999.0, m

    interruptions = result.get("interruptions", [])
    nodules = result.get("nodules", [])
    components = result.get("components", [])

    coverage = m["coverage"]
    continuity = m["continuity"]
    height_frac = m["height_frac"]
    area_frac = m["area_frac"]
    width_frac = m["width_frac"]
    y_center_frac = m["y_center_frac"]
    thickness_var_norm = m["thickness_var_norm"]
    components_count = m["components_count"]
    slope = m["slope"]

    score = 0.0
    reasons = []

    # Pleura trebuie sa fie extinsa orizontal.
    score += 4.0 * min(1.0, coverage / 0.55)
    score += 2.0 * min(1.0, width_frac / 0.65)
    score += 1.2 * min(1.0, continuity)

    # Pozitie verticala plauzibila.
    if 0.10 <= y_center_frac <= 0.85:
        score += 1.0
    else:
        score -= 2.0
        reasons.append("bad_vertical_position")

    # Limite diferite in functie de ipoteza de lucru.
    if mode == "surplus":
        max_height = 0.28
        max_area = 0.16
    elif mode == "pathologic":
        max_height = 0.62
        max_area = 0.28
    else:
        max_height = 0.42
        max_area = 0.22

    if height_frac <= max_height:
        score += 1.0
    else:
        score -= 5.0 * (height_frac - max_height)
        reasons.append("too_tall")

    if area_frac <= max_area:
        score += 1.0
    else:
        score -= 6.0 * (area_frac - max_area)
        reasons.append("too_large_area")

    # Penalizare pentru surplus.
    if mode == "surplus":
        score -= 3.5 * area_frac
        score -= 1.8 * height_frac
    else:
        score -= 1.5 * area_frac

    # Penalizare componente multiple. Pathologic permite mai multe bucati.
    if mode == "pathologic":
        score -= 0.08 * max(0, components_count - 1)
    elif mode == "surplus":
        score -= 0.35 * max(0, components_count - 1)
    else:
        score -= 0.20 * max(0, components_count - 1)

    # Penalizare variatie de grosime. Pathologic este mai permisiv.
    if mode == "pathologic":
        score -= 0.30 * min(3.0, thickness_var_norm)
    else:
        score -= 0.85 * min(3.0, thickness_var_norm)

    # Panta mare poate indica artefact sau contur gresit.
    if mode == "pathologic":
        max_slope = 2.2
    else:
        max_slope = 1.4

    if slope > max_slope:
        score -= 1.5
        reasons.append("large_slope")

    # Bonus mic pentru cazuri patologice daca apar semnale morfologice.
    if mode == "pathologic":
        if len(interruptions) > 0:
            score += 0.15 * min(4, len(interruptions))

        if len(nodules) > 0:
            score += 0.10 * min(4, len(nodules))

    accepted = True

    if coverage < 0.06:
        accepted = False
        reasons.append("low_coverage")

    if width_frac < 0.06:
        accepted = False
        reasons.append("low_width")

    if area_frac < 0.00005:
        accepted = False
        reasons.append("too_few_pixels")

    if height_frac > 0.70:
        accepted = False
        reasons.append("huge_height")

    if area_frac > 0.35:
        accepted = False
        reasons.append("huge_area")

    if score < 1.0:
        accepted = False
        reasons.append("low_score")

    suspect = False

    if not accepted:
        suspect = True

    if score < 2.2:
        suspect = True

    if mode != "pathologic" and components_count > 5:
        suspect = True

    if mode != "pathologic" and thickness_var_norm > 2.0:
        suspect = True

    if len(reasons) == 0:
        reason = "ok"
    else:
        reason = ",".join(sorted(set(reasons)))

    m["score"] = float(score)
    m["accepted"] = bool(accepted)
    m["suspect"] = bool(suspect)
    m["reason"] = reason
    m["mode"] = mode
    m["interruptions_count"] = len(interruptions)
    m["nodules_count"] = len(nodules)
    m["raw_components_count"] = len(components)

    return score, m


def RunAutomaticPleuraDetection(crop_Image_rgb, one_pixel=ONE_PIXEL_FALLBACK):
    candidates = []
    mode_errors = []

    for mode in AUTO_MODES:
        try:
            interpreted_Image = crop_Image_rgb.copy()

            result = ExtractPleuralLine(
                crop_Image_rgb,
                interpreted_Image,
                mode=mode,
                one_pixel=one_pixel
            )

            final_contours = result.get("final_contours", [])

            if final_contours is None or len(final_contours) == 0:
                raise ValueError("Nu s-a extras niciun contur final in modul " + mode)

            score, metrics = ScorePleuraResult(
                result,
                crop_Image_rgb.shape,
                mode
            )

            candidates.append({
                "mode": mode,
                "result": result,
                "score": score,
                "metrics": metrics
            })

        except Exception as e:
            mode_errors.append({
                "mode": mode,
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    if len(candidates) == 0:
        msg = "Toate modurile automate au esuat."

        for err in mode_errors:
            msg += "\n[" + err["mode"] + "] " + err["error"]

        raise ValueError(msg)

    best = max(candidates, key=lambda x: x["score"])

    return best, candidates, mode_errors


# ============================================================
# CONTOUR IDENTIFICATION
# ============================================================

def IdentifyPrincipalContour(img, deviation=50):
    points = []

    ExtractContour(img, points)

    if len(points) < 2:
        raise ValueError("[IdentifyPrincipalContour] Prea putine puncte candidate.")

    distances = removeOutliers(points, deviation)

    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 2:
        raise ValueError("[IdentifyPrincipalContour] Componenta principala prea mica.")

    poly_line, PX_fit = IdnetifyPoly(PX_, PY_, 3)

    pleural_underline = Fit(pts, poly_line, deviation)

    if len(pleural_underline) < 2:
        pleural_underline = pts

    if len(pleural_underline) < 2:
        raise ValueError("[IdentifyPrincipalContour] Pleura principala invalida.")

    for p in pleural_underline:
        x0 = int(p.x + 10)
        y0 = int(p.y)

        if 0 <= x0 < img.shape[0] and 0 <= y0 < img.shape[1]:
            img[x0:img.shape[0] - 1, y0:y0 + 1] = 0

    Xmin = int(min(p.x for p in pleural_underline)) - 10
    Ymin = int(min(p.y for p in pleural_underline)) - 10
    Xmax = int(max(p.x for p in pleural_underline)) + 10
    Ymax = int(max(p.y for p in pleural_underline)) + 10

    Xmin = max(0, Xmin)
    Ymin = max(0, Ymin)
    Xmax = min(img.shape[0] - 1, Xmax)
    Ymax = min(img.shape[1] - 1, Ymax)

    if Xmax <= Xmin or Ymax <= Ymin:
        raise ValueError("[IdentifyPrincipalContour] ROI principala invalida.")

    ROI = img[Xmin:Xmax, Ymin:Ymax]

    contours = ConnectedContour(ROI)

    PX = []
    PY = []
    pts = []

    class LocalPoint:
        x = 0.0
        y = 0.0

    if contours is not False:
        for k in contours:
            for i in k:
                for j in i:
                    p = LocalPoint()
                    p.x = j[1] + Xmin
                    p.y = j[0] + Ymin
                    pts.append(p)
                    PX.append(p.x)
                    PY.append(p.y)

    if len(pts) < 2:
        pts = pleural_underline
        PX = [p.x for p in pts]
        PY = [p.y for p in pts]

    pleura, PX_, PY_ = Fit2(pts, poly_line, 50, PX, PY)

    if len(pleura) < 2:
        pleura = pts
        PX_ = [p.x for p in pleura]
        PY_ = [p.y for p in pleura]

    poly_line2, PX_fit2 = IdnetifyPoly(PX_, PY_, 1)

    if len(PX_fit2) > 0:
        mean_x = float(np.mean(PX_fit2))
    else:
        mean_x = 999

    retry = mean_x < 200

    ps = []

    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)

    return ctr, PY_, poly_line2, pleura, retry


def IdentifySecondaryContour(lateral_poly, img, minX, minY):
    points = []

    ExtractContour(img, points)

    if len(points) < 2:
        return False, False, False, False, 1

    distances = removeOutliers(points, 50)

    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 2:
        return False, False, False, False, 1

    poly_line, PX_fit = IdnetifyPoly(PX_, PY_, 3)

    deviation = 30
    pleural_underline = Fit(pts, poly_line, deviation)

    if len(pleural_underline) < 2:
        return False, False, False, False, 1

    for p in pleural_underline:
        x0 = int(p.x + 10)
        y0 = int(p.y)

        if 0 <= x0 < img.shape[0] and 0 <= y0 < img.shape[1]:
            img[x0:img.shape[0] - 1, y0:y0 + 1] = 0

    Xmin = int(min(p.x for p in pleural_underline)) - 20
    Ymin = int(min(p.y for p in pleural_underline)) - 10
    Xmax = int(max(p.x for p in pleural_underline)) + 10
    Ymax = int(max(p.y for p in pleural_underline)) + 10

    Xmin = max(0, Xmin)
    Ymin = max(0, Ymin)
    Xmax = min(img.shape[0] - 1, Xmax)
    Ymax = min(img.shape[1] - 1, Ymax)

    if Xmax <= Xmin or Ymax <= Ymin:
        return False, False, False, False, 1

    ROI = img[Xmin:Xmax, Ymin:Ymax]

    contours = ConnectedContour(ROI)

    if contours is False:
        return False, False, False, False, 1

    PX = []
    PY = []
    pts = []

    class LocalPoint:
        x = 0.0
        y = 0.0

    for k in contours:
        for i in k:
            for j in i:
                p = LocalPoint()

                p.x = j[1] + Xmin + minX
                p.y = j[0] + Ymin + minY

                pts.append(p)

                PX.append(p.x)
                PY.append(p.y)

    deviation = 100
    pleura, PX_, PY_ = Fit2(pts, lateral_poly, deviation, PX, PY)

    if len(pleura) < 2:
        return False, False, False, False, 1

    ps = []

    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)

    return ctr, PY_, PX_, pleura, 0


# ============================================================
# COMPONENT GEOMETRY
# ============================================================

def GetComponentBounds(component):
    stats = ComponentStats(component)

    if stats is None:
        return None

    return {
        "min_x": stats["min_x"],
        "max_x": stats["max_x"],
        "min_y": stats["min_y"],
        "max_y": stats["max_y"],
        "center_x": int(stats["center_col"]),
        "center_y": int(stats["center_row"])
    }


def GetEndpoint(component, side):
    pts = component.reshape(-1, 2)

    if len(pts) == 0:
        return None

    xs = pts[:, 0]

    if side == "left":
        target_x = np.min(xs)
    else:
        target_x = np.max(xs)

    idx = np.where(xs == target_x)[0]

    if len(idx) == 0:
        return None

    selected = pts[idx]
    y = int(np.median(selected[:, 1]))
    x = int(target_x)

    return x, y


def component_column_profile(component):
    if component is False or component is None:
        return None

    pts = component.reshape(-1, 2)

    if len(pts) < 2:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]

    unique_x = np.unique(xs)

    prof_x = []
    top = []
    bottom = []
    center = []
    thick = []

    for x in unique_x:
        y_vals = ys[xs == x]

        if len(y_vals) == 0:
            continue

        y1 = int(np.min(y_vals))
        y2 = int(np.max(y_vals))

        prof_x.append(int(x))
        top.append(y1)
        bottom.append(y2)
        center.append(int(round((y1 + y2) / 2.0)))
        thick.append(int(y2 - y1 + 1))

    if len(prof_x) < 2:
        return None

    return {
        "x": np.array(prof_x, dtype=np.int32),
        "top": np.array(top, dtype=np.int32),
        "bottom": np.array(bottom, dtype=np.int32),
        "center": np.array(center, dtype=np.int32),
        "thickness": np.array(thick, dtype=np.int32)
    }


# ============================================================
# PATHOLOGY CANDIDATES
# ============================================================

def DetectInterruptionCandidates(components, image_shape, one_pixel, mode="normal"):
    h, w = image_shape[:2]

    if len(components) < 2:
        return []

    valid_components = []

    for comp in components:
        bounds = GetComponentBounds(comp)

        if bounds is not None:
            valid_components.append(comp)

    valid_components = sorted(
        valid_components,
        key=lambda c: GetComponentBounds(c)["min_x"]
    )

    candidates = []

    min_gap_px = max(4, int(round(0.25 / max(one_pixel, 1e-6))))
    max_gap_px = max(14, int(round(2.50 / max(one_pixel, 1e-6))))

    max_vertical_diff = max(25, int(0.10 * h))

    for i in range(len(valid_components) - 1):
        left_comp = valid_components[i]
        right_comp = valid_components[i + 1]

        left_end = GetEndpoint(left_comp, "right")
        right_start = GetEndpoint(right_comp, "left")

        if left_end is None or right_start is None:
            continue

        x1, y1 = left_end
        x2, y2 = right_start

        gap = int(x2 - x1)
        vertical_diff = int(abs(y2 - y1))

        if gap <= 0:
            continue

        if gap < min_gap_px:
            continue

        if gap > max_gap_px:
            continue

        if vertical_diff > max_vertical_diff:
            continue

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        candidates.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "gap_px": gap,
            "gap_mm": gap * one_pixel,
            "vertical_diff": vertical_diff
        })

    return candidates


def DetectNoduleCandidates(components, image_shape, one_pixel, mode="normal"):
    h, w = image_shape[:2]

    candidates = []

    for comp_idx, comp in enumerate(components):
        profile = component_column_profile(comp)

        if profile is None:
            continue

        xs = profile["x"]
        top = profile["top"]
        bottom = profile["bottom"]
        thickness = profile["thickness"]

        if len(thickness) < 8:
            continue

        med_thick = float(np.median(thickness))

        if med_thick <= 0:
            continue

        if mode == "pathologic":
            thick_thr = max(med_thick * 1.55, med_thick + 5)
        else:
            thick_thr = max(med_thick * 1.85, med_thick + 7)

        strong = thickness >= thick_thr

        i = 0

        while i < len(strong):
            if not strong[i]:
                i += 1
                continue

            start = i

            while i < len(strong) and strong[i]:
                i += 1

            end = i - 1

            seg_w_px = int(xs[end] - xs[start] + 1)
            seg_w_mm = seg_w_px * one_pixel

            min_w_px = max(3, int(round(0.20 / max(one_pixel, 1e-6))))
            max_w_px = max(18, int(round(5.00 / max(one_pixel, 1e-6))))

            if seg_w_px < min_w_px:
                continue

            if seg_w_px > max_w_px:
                continue

            x1 = int(xs[start])
            x2 = int(xs[end])

            y1 = int(np.min(top[start:end + 1]))
            y2 = int(np.max(bottom[start:end + 1]))

            pad_x = max(3, int(round(0.25 / max(one_pixel, 1e-6))))
            pad_y = max(3, int(round(0.25 / max(one_pixel, 1e-6))))

            x1p = max(0, x1 - pad_x)
            x2p = min(w - 1, x2 + pad_x)
            y1p = max(0, y1 - pad_y)
            y2p = min(h - 1, y2 + pad_y)

            candidates.append({
                "component": comp_idx,
                "x1": x1p,
                "y1": y1p,
                "x2": x2p,
                "y2": y2p,
                "cx": int(round((x1p + x2p) / 2.0)),
                "cy": int(round((y1p + y2p) / 2.0)),
                "width_px": seg_w_px,
                "width_mm": seg_w_mm,
                "median_thickness": med_thick,
                "max_thickness": int(np.max(thickness[start:end + 1]))
            })

    return candidates


# ============================================================
# MASK BUILDING
# ============================================================

def BuildComponentMask(components, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for comp in components:
        if comp is False or comp is None:
            continue

        if len(comp) < 2:
            continue

        cv2.drawContours(mask, [comp], -1, 255, -1)

    return mask


def BuildDisplayPleuraMask(components, image_shape, mode="normal"):
    h, w = image_shape[:2]
    mask = BuildComponentMask(components, image_shape)

    valid_components = []

    for comp in components:
        bounds = GetComponentBounds(comp)

        if bounds is not None:
            valid_components.append(comp)

    if len(valid_components) <= 1:
        return mask

    valid_components = sorted(
        valid_components,
        key=lambda c: GetComponentBounds(c)["min_x"]
    )

    if mode == "surplus":
        max_gap = max(20, int(0.035 * w))
        max_vertical_diff = max(18, int(0.035 * h))
        connection_thickness = 2

    elif mode == "pathologic":
        # Nu unim agresiv: golurile pot fi intreruperi reale.
        max_gap = max(18, int(0.030 * w))
        max_vertical_diff = max(28, int(0.060 * h))
        connection_thickness = 2

    else:
        max_gap = max(45, int(0.080 * w))
        max_vertical_diff = max(35, int(0.080 * h))
        connection_thickness = 4

    for i in range(len(valid_components) - 1):
        left_comp = valid_components[i]
        right_comp = valid_components[i + 1]

        left_end = GetEndpoint(left_comp, "right")
        right_start = GetEndpoint(right_comp, "left")

        if left_end is None or right_start is None:
            continue

        x1, y1 = left_end
        x2, y2 = right_start

        gap = x2 - x1
        vertical_diff = abs(y2 - y1)

        if gap <= 0:
            continue

        if gap <= max_gap and vertical_diff <= max_vertical_diff:
            cv2.line(
                mask,
                (x1, y1),
                (x2, y2),
                255,
                thickness=connection_thickness
            )

    return mask


# ============================================================
# FINAL PLEURAL EXTRACTION
# ============================================================

def ExtractPleuralLine(orig_Image, interpreted_Image, mode="normal", one_pixel=ONE_PIXEL_FALLBACK):
    img = orig_Image.copy()

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    nr_of_colors = 20
    img = ReduceColorPalette(img, nr_of_colors)

    # Pragul vechi a mers mai bine pe setul actual.
    # Pentru moment nu schimbam binarizarea; ne axam doar pe identificare.
    tresh = np.max(img) - 10
    img = Binarize(img, tresh)
    img = (1 - img).astype(np.uint8)

    left_contour_components = []
    right_contour_components = []
    accepted_components = []

    PrincipalComponent, PYS, poly_line2, points, retry = IdentifyPrincipalContour(img, 50)

    principal_ok = IsPrincipalComponentPlausible(
        PrincipalComponent,
        image_shape=img.shape
    )

    if not principal_ok:
        print("  [WARN] Componenta principala pare suspecta, dar continui.")

    accepted_components.append(PrincipalComponent)

    cv2.drawContours(
        interpreted_Image,
        [PrincipalComponent],
        0,
        (0, 255, 0),
        1
    )

    # -------------------------
    # LEFT SIDE
    # -------------------------
    try:
        Ymin = int(min(PYS) + 10)
    except Exception:
        Ymin = 0

    Ymin = max(0, min(Ymin, img.shape[1] - 1))

    Ex = []
    Ey = []

    for y in range(0, Ymin):
        Ey.append(y)
        Ex.append(poly_line2(y))

    if len(Ex) > 2:
        minX = int(min(Ex)) - 20
        minX = max(0, minX)

        maxX = img.shape[0] - 1
        left_poly_line, _ = IdnetifyPoly(Ex, Ey, 3)

        newROI = img[minX:maxX, 0:Ymin]

        isEmpty = 0
        last_Y = Ymin + 5

        while Ymin < last_Y and newROI.shape[1] > 20 and isEmpty == 0:
            if len(newROI) > 0:
                NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(
                    left_poly_line,
                    newROI,
                    minX,
                    0
                )
            else:
                break

            if isEmpty != 1:
                is_good = IsPleuraLikeComponent(
                    NextComponent,
                    principal_poly=poly_line2,
                    image_shape=img.shape,
                    mode=mode
                )

                if not is_good:
                    isEmpty = 1
                    break

                cv2.drawContours(
                    interpreted_Image,
                    [NextComponent],
                    0,
                    (50, 150, 255),
                    1
                )

                left_contour_components.append(NextComponent)
                accepted_components.append(NextComponent)

                last_Y = Ymin

                try:
                    Ymin = int(min(PY_) + 10)
                    minX = int(min(PX_) - 10)
                except Exception:
                    break

                minX = max(0, minX)
                Ymin = max(0, min(Ymin, img.shape[1] - 1))

                newROI = img[minX:maxX, 0:Ymin]

            else:
                break

    # -------------------------
    # RIGHT SIDE
    # -------------------------
    try:
        Ymax = int(max(PYS) - 10)
    except Exception:
        Ymax = img.shape[1] - 1

    Ymax = max(0, min(Ymax, img.shape[1] - 1))

    Ex = []
    Ey = []

    for y in range(Ymax, img.shape[1] - 1):
        Ey.append(y)
        Ex.append(poly_line2(y))

    if len(Ex) > 2:
        minX = int(min(Ex)) - 10
        minX = max(0, minX)

        maxX = img.shape[0] - 1
        right_poly_line, _ = IdnetifyPoly(Ex, Ey, 3)

        newROI = img[minX:maxX, Ymax:img.shape[1] - 1]

        last_Y = Ymax - 5
        isEmpty = 0

        while Ymax > last_Y and newROI.shape[1] > 20 and isEmpty == 0:
            if len(newROI) > 0:
                NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(
                    right_poly_line,
                    newROI,
                    minX,
                    Ymax
                )
            else:
                break

            if isEmpty != 1:
                is_good = IsPleuraLikeComponent(
                    NextComponent,
                    principal_poly=poly_line2,
                    image_shape=img.shape,
                    mode=mode
                )

                if not is_good:
                    isEmpty = 1
                    break

                cv2.drawContours(
                    interpreted_Image,
                    [NextComponent],
                    0,
                    (255, 0, 0),
                    1
                )

                right_contour_components.append(NextComponent)
                accepted_components.append(NextComponent)

                last_Y = Ymax

                try:
                    Ymax = int(max(PY_))
                    minX = int(min(PX_) - 10)
                except Exception:
                    break

                minX = max(0, minX)
                Ymax = max(0, min(Ymax, img.shape[1] - 1))

                newROI = img[minX:maxX, Ymax:img.shape[1] - 1]

            else:
                break

    component_mask = BuildComponentMask(
        accepted_components,
        interpreted_Image.shape[:2]
    )

    display_mask = BuildDisplayPleuraMask(
        accepted_components,
        interpreted_Image.shape[:2],
        mode=mode
    )


    final_contours = FindAllContoursFromMask(display_mask)

    if len(final_contours) == 0:
        final_contours = accepted_components

    interruptions = DetectInterruptionCandidates(
        accepted_components,
        interpreted_Image.shape[:2],
        one_pixel,
        mode=mode
    )

    nodules = DetectNoduleCandidates(
        accepted_components,
        interpreted_Image.shape[:2],
        one_pixel,
        mode=mode
    )

    return {
        "final_contours": final_contours,
        "components": accepted_components,
        "component_mask": component_mask,
        "display_mask": display_mask,
        "orig_image": orig_Image,
        "interruptions": interruptions,
        "nodules": nodules
    }


# ============================================================
# OUTPUT HELPERS
# ============================================================

def draw_components_overlay(image_rgb, components):
    out = image_rgb.copy()

    for comp in components:
        try:
            cv2.drawContours(out, [comp], -1, (255, 0, 0), 1)
        except Exception:
            pass

    return out


def draw_final_overlay(image_rgb, final_contours):
    out = image_rgb.copy()

    for ctr in final_contours:
        try:
            cv2.drawContours(out, [ctr], -1, (0, 255, 0), 1)
        except Exception:
            pass

    return out


def draw_diagnostic_overlay(image_rgb, final_contours, interruptions, nodules):
    out = image_rgb.copy()

    for ctr in final_contours:
        try:
            cv2.drawContours(out, [ctr], -1, (0, 255, 0), 1)
        except Exception:
            pass

    for item in interruptions:
        cx = int(item["cx"])
        cy = int(item["cy"])

        r = max(5, int(item["gap_px"] / 2))

        cv2.circle(
            out,
            (cx, cy),
            r,
            (255, 0, 0),
            2
        )

        cv2.line(
            out,
            (int(item["x1"]), int(item["y1"])),
            (int(item["x2"]), int(item["y2"])),
            (255, 0, 0),
            1
        )

    for item in nodules:
        cv2.rectangle(
            out,
            (int(item["x1"]), int(item["y1"])),
            (int(item["x2"]), int(item["y2"])),
            (255, 255, 0),
            2
        )

    return out


def build_interruption_mask(shape, interruptions):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for item in interruptions:
        cx = int(item["cx"])
        cy = int(item["cy"])
        r = max(5, int(item["gap_px"] / 2))

        cv2.circle(mask, (cx, cy), r, 255, 2)

    return mask


def build_nodule_mask(shape, nodules):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for item in nodules:
        cv2.rectangle(
            mask,
            (int(item["x1"]), int(item["y1"])),
            (int(item["x2"]), int(item["y2"])),
            255,
            2
        )

    return mask


def make_contact_sheet(image_folder, output_path, cols=8, thumb_w=260, thumb_h=180):
    files = []

    for name in os.listdir(image_folder):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(name)

    files = sorted(files)

    if len(files) == 0:
        print("[CONTACT_SHEET] Nu exista imagini in:", image_folder)
        return

    rows = int(np.ceil(len(files) / cols))

    sheet_w = cols * thumb_w
    sheet_h = rows * thumb_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, filename in enumerate(files):
        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (thumb_w, thumb_h - 28))

        r = idx // cols
        c = idx % cols

        y1 = r * thumb_h
        x1 = c * thumb_w

        sheet[y1:y1 + thumb_h - 28, x1:x1 + thumb_w] = img

        label = filename
        label = label.replace("_detected_pleura.png", "")
        label = label.replace("_diagnostic.png", "")
        label = label.replace("_components.png", "")
        label = label.replace("_mask.png", "")
        label = label.replace("_interruptions.png", "")
        label = label.replace("_nodules.png", "")
        label = label.replace("_suspect.png", "")

        cv2.putText(
            sheet,
            label,
            (x1 + 8, y1 + thumb_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, sheet)
    print("[CONTACT_SHEET] Salvat:", output_path)


def write_metrics_file(path, metrics):
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(metrics.keys()):
            f.write(str(key) + "=" + str(metrics[key]) + "\n")


def write_auto_mode_csv(path, rows):
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    keys = [
        "image",
        "chosen_mode",
        "score",
        "accepted",
        "suspect",
        "reason",
        "coverage",
        "width_frac",
        "height_frac",
        "area_frac",
        "continuity",
        "y_center_frac",
        "median_thickness",
        "thickness_iqr",
        "thickness_var_norm",
        "components_count",
        "interruptions_count",
        "nodules_count"
    ]

    def esc(value):
        s = "" if value is None else str(value)
        if "," in s or "\n" in s or '"' in s:
            s = '"' + s.replace('"', '""') + '"'
        return s

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(esc(row.get(k, "")) for k in keys) + "\n")


# ============================================================
# MAIN BATCH
# ============================================================

def main():
    all_contours_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_CONTOURS"))
    all_diagnostic_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_DIAGNOSTIC"))
    all_components_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_COMPONENTS"))
    all_masks_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_MASKS"))
    all_interruption_masks_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_INTERRUPTION_MASKS"))
    all_nodule_masks_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_NODULE_MASKS"))
    all_mode_results_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_MODE_RESULTS"))
    suspect_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "SUSPECT_RESULTS"))
    error_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ERRORS"))

    success_count = 0
    fail_count = 0
    suspect_count = 0

    failed_images = []
    suspect_images = []

    interruption_report = []
    nodule_report = []
    auto_mode_report = []

    print("\n" + "=" * 70)
    print("BATCH PLEURA - MOD AUTOMAT FARA HARDCODARE")
    print("Input :", INPUT_DIR)
    print("Output contururi:", all_contours_dir)
    print("Output diagnostic:", all_diagnostic_dir)
    print("Output toate modurile:", all_mode_results_dir)
    print("Moduri automate:", AUTO_MODES)
    print("=" * 70 + "\n")

    for idx in range(START_IDX, END_IDX + 1):
        img_name = str(idx) + ".jpg"
        img_path = os.path.join(INPUT_DIR, img_name)

        print("\n[" + str(idx) + "/" + str(END_IDX) + "] Procesez:", img_name)

        if not os.path.exists(img_path):
            print("  [SKIP] Nu exista fisierul:", img_path)
            fail_count += 1
            failed_images.append((img_name, "Fisier inexistent"))
            continue

        try:
            orig_Image = io.imread(img_path)

            if orig_Image.ndim == 2:
                orig_Image = cv2.cvtColor(orig_Image, cv2.COLOR_GRAY2RGB)
            elif orig_Image.ndim == 3 and orig_Image.shape[2] == 4:
                orig_Image = cv2.cvtColor(orig_Image, cv2.COLOR_RGBA2RGB)

            try:
                one_pixel = PixelConverter(orig_Image)
            except Exception:
                one_pixel = ONE_PIXEL_FALLBACK

            crop_Image = CropBorder(orig_Image)

            if crop_Image.ndim == 3:
                crop_Image = cv2.cvtColor(crop_Image, cv2.COLOR_RGB2GRAY)

            crop_Image_rgb = cv2.cvtColor(crop_Image, cv2.COLOR_GRAY2RGB)

            best_candidate, all_candidates, mode_errors = RunAutomaticPleuraDetection(
                crop_Image_rgb,
                one_pixel=one_pixel
            )

            mode = best_candidate["mode"]
            result = best_candidate["result"]
            auto_score = best_candidate["score"]
            auto_metrics = best_candidate["metrics"]

            final_contours = result["final_contours"]
            components = result["components"]
            display_mask = result["display_mask"]
            component_mask = result["component_mask"]
            interruptions = result["interruptions"]
            nodules = result["nodules"]

            if final_contours is None or len(final_contours) == 0:
                raise ValueError("Nu s-a extras niciun contur final.")

            # ------------------------------------------------
            # Salvam rezultatele tuturor modurilor pentru audit.
            # ------------------------------------------------
            img_modes_dir = ensure_dir(
                os.path.join(
                    all_mode_results_dir,
                    format(idx, "02d")
                )
            )

            for cand in all_candidates:
                cand_mode = cand["mode"]
                cand_result = cand["result"]
                cand_metrics = cand["metrics"]

                cand_dir = ensure_dir(
                    os.path.join(
                        img_modes_dir,
                        cand_mode
                    )
                )

                cand_overlay = draw_final_overlay(
                    crop_Image_rgb,
                    cand_result["final_contours"]
                )

                cand_diagnostic = draw_diagnostic_overlay(
                    crop_Image_rgb,
                    cand_result["final_contours"],
                    cand_result["interruptions"],
                    cand_result["nodules"]
                )

                cand_components_overlay = draw_components_overlay(
                    crop_Image_rgb,
                    cand_result["components"]
                )

                save_rgb(
                    os.path.join(cand_dir, "overlay.png"),
                    cand_overlay
                )

                save_rgb(
                    os.path.join(cand_dir, "diagnostic.png"),
                    cand_diagnostic
                )

                save_rgb(
                    os.path.join(cand_dir, "components.png"),
                    cand_components_overlay
                )

                cv2.imwrite(
                    os.path.join(cand_dir, "mask.png"),
                    cand_result["display_mask"]
                )

                cv2.imwrite(
                    os.path.join(cand_dir, "component_mask.png"),
                    cand_result["component_mask"]
                )

                write_metrics_file(
                    os.path.join(cand_dir, "metrics.txt"),
                    cand_metrics
                )

            if len(mode_errors) > 0:
                with open(os.path.join(img_modes_dir, "mode_errors.txt"), "w", encoding="utf-8") as f:
                    for err in mode_errors:
                        f.write("MODE: " + str(err["mode"]) + "\n")
                        f.write("ERROR: " + str(err["error"]) + "\n")
                        f.write(err["traceback"])
                        f.write("\n" + "=" * 60 + "\n")

            # -----------------------------
            # Overlay simplu contur
            # -----------------------------
            result_overlay = draw_final_overlay(
                crop_Image_rgb,
                final_contours
            )

            output_path = os.path.join(
                all_contours_dir,
                format(idx, "02d") + "_detected_pleura.png"
            )

            save_rgb(output_path, result_overlay)

            # -----------------------------
            # Overlay diagnostic:
            # verde = pleura
            # rosu/albastru = intreruperi
            # galben = noduli
            # -----------------------------
            diagnostic_overlay = draw_diagnostic_overlay(
                crop_Image_rgb,
                final_contours,
                interruptions,
                nodules
            )

            diagnostic_path = os.path.join(
                all_diagnostic_dir,
                format(idx, "02d") + "_diagnostic.png"
            )

            save_rgb(diagnostic_path, diagnostic_overlay)

            # -----------------------------
            # Componente individuale
            # -----------------------------
            components_overlay = draw_components_overlay(
                crop_Image_rgb,
                components
            )

            components_path = os.path.join(
                all_components_dir,
                format(idx, "02d") + "_components.png"
            )

            save_rgb(components_path, components_overlay)

            # -----------------------------
            # Masca finala
            # -----------------------------
            mask_path = os.path.join(
                all_masks_dir,
                format(idx, "02d") + "_mask.png"
            )

            cv2.imwrite(mask_path, display_mask)

            # -----------------------------
            # Masca intreruperi
            # -----------------------------
            interruption_mask = build_interruption_mask(
                crop_Image.shape,
                interruptions
            )

            interruption_mask_path = os.path.join(
                all_interruption_masks_dir,
                format(idx, "02d") + "_interruptions.png"
            )

            cv2.imwrite(interruption_mask_path, interruption_mask)

            # -----------------------------
            # Masca noduli
            # -----------------------------
            nodule_mask = build_nodule_mask(
                crop_Image.shape,
                nodules
            )

            nodule_mask_path = os.path.join(
                all_nodule_masks_dir,
                format(idx, "02d") + "_nodules.png"
            )

            cv2.imwrite(nodule_mask_path, nodule_mask)

            # ------------------------------------------------
            # Decizia de suspect nu mai depinde de imagine wrong.
            # Este complet automata, pe baza scorului si metricilor.
            # ------------------------------------------------
            is_plausible = bool(auto_metrics["accepted"])

            if auto_metrics["suspect"]:
                is_plausible = False

            if not is_plausible:
                suspect_count += 1
                suspect_images.append((img_name, "Contur suspect automat: " + str(auto_metrics["reason"])))

                suspect_path = os.path.join(
                    suspect_dir,
                    format(idx, "02d") + "_suspect.png"
                )

                save_rgb(suspect_path, diagnostic_overlay)

                print("  [WARN] Contur suspect, salvat si in:", suspect_path)

            for item in interruptions:
                interruption_report.append(
                    {
                        "image": img_name,
                        "gap_px": item["gap_px"],
                        "gap_mm": item["gap_mm"],
                        "cx": item["cx"],
                        "cy": item["cy"]
                    }
                )

            for item in nodules:
                nodule_report.append(
                    {
                        "image": img_name,
                        "width_px": item["width_px"],
                        "width_mm": item["width_mm"],
                        "cx": item["cx"],
                        "cy": item["cy"],
                        "max_thickness": item["max_thickness"]
                    }
                )

            auto_mode_report.append(
                {
                    "image": img_name,
                    "chosen_mode": mode,
                    "score": auto_score,
                    "accepted": auto_metrics["accepted"],
                    "suspect": auto_metrics["suspect"],
                    "reason": auto_metrics["reason"],
                    "coverage": auto_metrics["coverage"],
                    "width_frac": auto_metrics["width_frac"],
                    "height_frac": auto_metrics["height_frac"],
                    "area_frac": auto_metrics["area_frac"],
                    "continuity": auto_metrics["continuity"],
                    "y_center_frac": auto_metrics["y_center_frac"],
                    "median_thickness": auto_metrics["median_thickness"],
                    "thickness_iqr": auto_metrics["thickness_iqr"],
                    "thickness_var_norm": auto_metrics["thickness_var_norm"],
                    "components_count": auto_metrics["components_count"],
                    "interruptions_count": auto_metrics["interruptions_count"],
                    "nodules_count": auto_metrics["nodules_count"]
                }
            )

            success_count += 1

            print("  [OK] Salvat:", output_path)
            print("  one_pixel =", one_pixel)
            print("  components =", len(components))
            print("  final_contours =", len(final_contours))
            print("  intreruperi =", len(interruptions))
            print("  noduli =", len(nodules))
            print("  mode ales automat =", mode)
            print("  auto_score =", auto_score)
            print("  auto_reason =", auto_metrics["reason"])
            print("  coverage =", auto_metrics["coverage"])
            print("  area_frac =", auto_metrics["area_frac"])
            print("  height_frac =", auto_metrics["height_frac"])

        except Exception as e:
            fail_count += 1
            failed_images.append((img_name, str(e)))

            print("  [ESEC]", img_name + ":", e)

            error_path = os.path.join(
                error_dir,
                format(idx, "02d") + "_error.txt"
            )

            with open(error_path, "w", encoding="utf-8") as f:
                f.write("Imagine: " + img_name + "\n")
                f.write("Eroare: " + str(e) + "\n\n")
                f.write(traceback.format_exc())

            continue

    # ========================================================
    # CONTACT SHEETS
    # ========================================================

    contact_sheet_path = os.path.join(OUTPUT_ROOT, "contact_sheet_ALL_CONTOURS.png")

    make_contact_sheet(
        all_contours_dir,
        contact_sheet_path,
        cols=8,
        thumb_w=260,
        thumb_h=180
    )

    diagnostic_sheet_path = os.path.join(OUTPUT_ROOT, "contact_sheet_ALL_DIAGNOSTIC.png")

    make_contact_sheet(
        all_diagnostic_dir,
        diagnostic_sheet_path,
        cols=8,
        thumb_w=260,
        thumb_h=180
    )

    components_sheet_path = os.path.join(OUTPUT_ROOT, "contact_sheet_ALL_COMPONENTS.png")

    make_contact_sheet(
        all_components_dir,
        components_sheet_path,
        cols=8,
        thumb_w=260,
        thumb_h=180
    )

    suspect_sheet_path = os.path.join(OUTPUT_ROOT, "contact_sheet_SUSPECT_RESULTS.png")

    make_contact_sheet(
        suspect_dir,
        suspect_sheet_path,
        cols=8,
        thumb_w=260,
        thumb_h=180
    )

    # ========================================================
    # REPORTS
    # ========================================================

    summary_path = os.path.join(OUTPUT_ROOT, "summary.txt")
    interruption_report_path = os.path.join(OUTPUT_ROOT, "interruptions_report.txt")
    nodule_report_path = os.path.join(OUTPUT_ROOT, "nodules_report.txt")
    auto_mode_report_path = os.path.join(OUTPUT_ROOT, "auto_mode_report.txt")
    auto_mode_csv_path = os.path.join(OUTPUT_ROOT, "auto_mode_report.csv")

    write_auto_mode_csv(auto_mode_csv_path, auto_mode_report)

    with open(auto_mode_report_path, "w", encoding="utf-8") as f:
        f.write("MOD ALES AUTOMAT PER IMAGINE\n")
        f.write("=" * 70 + "\n")
        f.write("Moduri disponibile: " + str(AUTO_MODES) + "\n\n")

        if len(auto_mode_report) == 0:
            f.write("Nu exista rezultate automate.\n")
        else:
            for item in auto_mode_report:
                f.write(
                    "Imagine: "
                    + str(item["image"])
                    + " | mode="
                    + str(item["chosen_mode"])
                    + " | score={:.4f}".format(item["score"])
                    + " | accepted="
                    + str(item["accepted"])
                    + " | suspect="
                    + str(item["suspect"])
                    + " | reason="
                    + str(item["reason"])
                    + " | coverage={:.4f}".format(item["coverage"])
                    + " | area_frac={:.6f}".format(item["area_frac"])
                    + " | height_frac={:.4f}".format(item["height_frac"])
                    + " | components="
                    + str(item["components_count"])
                    + " | intreruperi="
                    + str(item["interruptions_count"])
                    + " | noduli="
                    + str(item["nodules_count"])
                    + "\n"
                )

    with open(interruption_report_path, "w", encoding="utf-8") as f:
        f.write("INTRERUPERI CANDIDATE\n")
        f.write("=" * 50 + "\n")

        if len(interruption_report) == 0:
            f.write("Nu s-au gasit intreruperi candidate.\n")
        else:
            for item in interruption_report:
                f.write(
                    "Imagine: "
                    + str(item["image"])
                    + " | gap_px="
                    + str(item["gap_px"])
                    + " | gap_mm={:.4f}".format(item["gap_mm"])
                    + " | center=("
                    + str(item["cx"])
                    + ", "
                    + str(item["cy"])
                    + ")\n"
                )

    with open(nodule_report_path, "w", encoding="utf-8") as f:
        f.write("NODULI CANDIDATI\n")
        f.write("=" * 50 + "\n")

        if len(nodule_report) == 0:
            f.write("Nu s-au gasit noduli candidati.\n")
        else:
            for item in nodule_report:
                f.write(
                    "Imagine: "
                    + str(item["image"])
                    + " | width_px="
                    + str(item["width_px"])
                    + " | width_mm={:.4f}".format(item["width_mm"])
                    + " | max_thickness="
                    + str(item["max_thickness"])
                    + " | center=("
                    + str(item["cx"])
                    + ", "
                    + str(item["cy"])
                    + ")\n"
                )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("BATCH PLEURA - MOD AUTOMAT FARA HARDCODARE\n")
        f.write("=" * 50 + "\n")
        f.write("Interval imagini: " + str(START_IDX) + "-" + str(END_IDX) + "\n")
        f.write("Reusite: " + str(success_count) + "\n")
        f.write("Esuate: " + str(fail_count) + "\n")
        f.write("Suspecte: " + str(suspect_count) + "\n")
        f.write("Total intreruperi candidate: " + str(len(interruption_report)) + "\n")
        f.write("Total noduli candidati: " + str(len(nodule_report)) + "\n\n")

        f.write("Moduri automate folosite:\n")
        f.write(str(AUTO_MODES) + "\n\n")

        f.write("Folder contururi:\n")
        f.write(all_contours_dir + "\n\n")

        f.write("Folder diagnostic:\n")
        f.write(all_diagnostic_dir + "\n\n")

        f.write("Folder componente:\n")
        f.write(all_components_dir + "\n\n")

        f.write("Folder masti:\n")
        f.write(all_masks_dir + "\n\n")

        f.write("Folder masti intreruperi:\n")
        f.write(all_interruption_masks_dir + "\n\n")

        f.write("Folder masti noduli:\n")
        f.write(all_nodule_masks_dir + "\n\n")

        f.write("Folder toate modurile:\n")
        f.write(all_mode_results_dir + "\n\n")

        f.write("Folder suspecte:\n")
        f.write(suspect_dir + "\n\n")

        f.write("Folder erori:\n")
        f.write(error_dir + "\n\n")

        f.write("Contact sheet contururi:\n")
        f.write(contact_sheet_path + "\n\n")

        f.write("Contact sheet diagnostic:\n")
        f.write(diagnostic_sheet_path + "\n\n")

        f.write("Contact sheet componente:\n")
        f.write(components_sheet_path + "\n\n")

        f.write("Contact sheet suspecte:\n")
        f.write(suspect_sheet_path + "\n\n")

        f.write("Raport mod automat TXT:\n")
        f.write(auto_mode_report_path + "\n\n")

        f.write("Raport mod automat CSV:\n")
        f.write(auto_mode_csv_path + "\n\n")

        f.write("Raport intreruperi:\n")
        f.write(interruption_report_path + "\n\n")

        f.write("Raport noduli:\n")
        f.write(nodule_report_path + "\n\n")

        f.write("MOD ALES AUTOMAT PER IMAGINE:\n")

        for item in auto_mode_report:
            f.write(
                "  "
                + str(item["image"])
                + " | mode="
                + str(item["chosen_mode"])
                + " | score={:.4f}".format(item["score"])
                + " | accepted="
                + str(item["accepted"])
                + " | suspect="
                + str(item["suspect"])
                + " | reason="
                + str(item["reason"])
                + " | coverage={:.4f}".format(item["coverage"])
                + " | area_frac={:.6f}".format(item["area_frac"])
                + " | height_frac={:.4f}".format(item["height_frac"])
                + " | components="
                + str(item["components_count"])
                + " | intreruperi="
                + str(item["interruptions_count"])
                + " | noduli="
                + str(item["nodules_count"])
                + "\n"
            )

        f.write("\n")

        if len(failed_images) > 0:
            f.write("Imagini esuate:\n")
            for name, err in failed_images:
                f.write("  " + name + ": " + err + "\n")

        if len(suspect_images) > 0:
            f.write("\nImagini suspecte:\n")
            for name, reason in suspect_images:
                f.write("  " + name + ": " + reason + "\n")

    print("\n" + "=" * 70)
    print("BATCH TERMINAT")
    print("Reusite :", success_count)
    print("Esuate  :", fail_count)
    print("Suspecte:", suspect_count)
    print("Total intreruperi candidate:", len(interruption_report))
    print("Total noduli candidati:", len(nodule_report))
    print("Contururi:", all_contours_dir)
    print("Diagnostic:", all_diagnostic_dir)
    print("Toate modurile:", all_mode_results_dir)
    print("Contact sheet diagnostic:", diagnostic_sheet_path)
    print("Raport automat CSV:", auto_mode_csv_path)
    print("Rezumat:", summary_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
