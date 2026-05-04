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
OUTPUT_ROOT = r"C:\Facultate\AN4\Licenta\Licenta-Cod\DEBUG_OUT_INITIAL_BATCH"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

START_IDX = 0
END_IDX = 61

ONE_PIXEL_FALLBACK = 0.07


# ============================================================
# CALIBRARE PE IMAGINILE VERIFICATE DE TINE
# ============================================================

GOOD_IDS = {
    2, 3, 4, 5, 6, 9, 10, 11, 12, 13,
    27, 29, 33, 35, 36, 41, 43, 50, 51
}

SURPLUS_IDS = {
    1, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 28, 32, 34, 47, 48, 49, 58
}

WRONG_IDS = {
    0
}

# Le tratăm ca pleură potențial patologică:
# poate avea întreruperi, bucăți separate, noduli, grosime locală mai mare.
PATHOLOGIC_IDS = {
    7, 8, 14, 15, 16, 30, 31, 37, 38, 39,
    40, 42, 44, 45, 46, 52, 53, 54, 55,
    56, 57, 59, 60, 61
}


def get_image_mode(idx):
    if idx in WRONG_IDS:
        return "wrong"

    if idx in SURPLUS_IDS:
        return "surplus"

    if idx in PATHOLOGIC_IDS:
        return "pathologic"

    return "normal"


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
        # Nu unim agresiv: golurile pot fi întreruperi reale.
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
    suspect_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "SUSPECT_RESULTS"))
    error_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ERRORS"))

    success_count = 0
    fail_count = 0
    suspect_count = 0

    failed_images = []
    suspect_images = []

    interruption_report = []
    nodule_report = []

    print("\n" + "=" * 70)
    print("BATCH PLEURA - CU NODULI SI INTRERUPERI")
    print("Input :", INPUT_DIR)
    print("Output contururi:", all_contours_dir)
    print("Output diagnostic:", all_diagnostic_dir)
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
            mode = get_image_mode(idx)

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
            interpreted_Image = crop_Image_rgb.copy()

            result = ExtractPleuralLine(
                crop_Image_rgb,
                interpreted_Image,
                mode=mode,
                one_pixel=one_pixel
            )

            final_contours = result["final_contours"]
            components = result["components"]
            display_mask = result["display_mask"]
            component_mask = result["component_mask"]
            interruptions = result["interruptions"]
            nodules = result["nodules"]

            if final_contours is None or len(final_contours) == 0:
                raise ValueError("Nu s-a extras niciun contur final.")

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

            is_plausible = FinalMaskLooksPlausible(
                display_mask,
                crop_Image.shape,
                mode=mode
            )

            if mode == "wrong":
                is_plausible = False

            if not is_plausible:
                suspect_count += 1
                suspect_images.append((img_name, "Contur suspect geometric sau imagine marcata wrong"))

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

            success_count += 1

            print("  [OK] Salvat:", output_path)
            print("  one_pixel =", one_pixel)
            print("  components =", len(components))
            print("  final_contours =", len(final_contours))
            print("  intreruperi =", len(interruptions))
            print("  noduli =", len(nodules))
            print("  mode =", mode)

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
        f.write("BATCH PLEURA - CU NODULI SI INTRERUPERI\n")
        f.write("=" * 50 + "\n")
        f.write("Interval imagini: " + str(START_IDX) + "-" + str(END_IDX) + "\n")
        f.write("Reusite: " + str(success_count) + "\n")
        f.write("Esuate: " + str(fail_count) + "\n")
        f.write("Suspecte: " + str(suspect_count) + "\n")
        f.write("Total intreruperi candidate: " + str(len(interruption_report)) + "\n")
        f.write("Total noduli candidati: " + str(len(nodule_report)) + "\n\n")

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

        f.write("Raport intreruperi:\n")
        f.write(interruption_report_path + "\n\n")

        f.write("Raport noduli:\n")
        f.write(nodule_report_path + "\n\n")

        f.write("GOOD_IDS:\n")
        f.write(str(sorted(GOOD_IDS)) + "\n\n")

        f.write("SURPLUS_IDS:\n")
        f.write(str(sorted(SURPLUS_IDS)) + "\n\n")

        f.write("PATHOLOGIC_IDS:\n")
        f.write(str(sorted(PATHOLOGIC_IDS)) + "\n\n")

        f.write("WRONG_IDS:\n")
        f.write(str(sorted(WRONG_IDS)) + "\n\n")

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
    print("Contact sheet diagnostic:", diagnostic_sheet_path)
    print("Rezumat:", summary_path)
    print("=" * 70)


if __name__ == "__main__":
    main()