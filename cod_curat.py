import os
import re
import math
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract as tes

from skimage import io
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_yen

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


INPUT_DIR = r"C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES"
OUTPUT_ROOT = r"C:\Facultate\AN4\Licenta\Licenta-Cod\DEBUG_OUT_AUTO_MODE"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

START_IDX = 0
END_IDX = 61

ONE_PIXEL_FALLBACK = 0.07

MAX_PIXEL_VALUE = 255
DEFAULT_BINARY_THRESHOLD = 128
BINARY_THRESHOLD = 110
MIN_BORDER_AREA = 1000
SMALL_CONTOUR_AREA = 30
PIXEL_CONTOUR_AREA = 20
LEFT_MARGIN_SEARCH_LIMIT = 100
DEFAULT_LEFT_BOUND = 25
CROP_RIGHT_MARGIN = 20
MIN_CROP_SIZE = 30
HORIZONTAL_KERNEL_SIZE = (2, 1)
MORPH_ITERATIONS = 3
PLEURA_COLOR_COUNT = 20
PLEURA_THRESHOLD_OFFSET = 10
DEFAULT_DEVIATION = 50
SECONDARY_INITIAL_DEVIATION = 30
SECONDARY_LATERAL_DEVIATION = 100
MAX_FIT_DEVIATION = 300


AUTO_MODES = ["normal", "surplus", "pathologic"]


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    dist: float = 0.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_rgb(path, img_rgb):
    if img_rgb.ndim == 2:
        cv2.imwrite(path, img_rgb)
    else:
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def crop_border(orig_Image):
    orig_img = orig_Image.copy()

    if orig_img.ndim == 2:
        gray_Image = orig_img.copy()
    else:
        gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

    _, img_bin = cv2.threshold(gray_Image, DEFAULT_BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)
    _, threshold = cv2.threshold(gray_Image, BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > MIN_BORDER_AREA:
            approx = cv2.approxPolyDP(
                cnt,
                0.01 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    _, threshold = cv2.threshold(gray_Image, BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    black = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > MIN_BORDER_AREA:
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
        last_small = white_pixels[1, white_pixels[1] < LEFT_MARGIN_SEARCH_LIMIT]

    if len(white_pixels[1]) == 0 or len(last_small) == 0:
        left_bound = DEFAULT_LEFT_BOUND
    else:
        left_bound = int(last_small[-1])

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < SMALL_CONTOUR_AREA:
            approx = cv2.approxPolyDP(
                cnt,
                0.00001 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    _, threshold = cv2.threshold(gray_Image, BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    black = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < SMALL_CONTOUR_AREA:
            approx = cv2.approxPolyDP(
                cnt,
                0.00001 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                cv2.drawContours(black, [approx], 0, 255, -1)

            if len(approx) == 2:
                cv2.drawContours(black, [approx], 0, 255, -1)

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HORIZONTAL_KERNEL_SIZE)

    img_temp2 = cv2.erode(black, hori_kernel, iterations=MORPH_ITERATIONS)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=MORPH_ITERATIONS)

    bar_image = horizontal_lines_img

    columns = np.zeros(bar_image.shape[1], dtype=int)

    for i in range(bar_image.shape[1]):
        columns[i] = np.count_nonzero(bar_image[:, i])

    if np.max(columns) == 0:
        return gray_Image.copy()

    bar_pos = np.where(columns == np.max(columns))[0][0]

    bar = bar_image[:, bar_pos] // 255
    bar_pixels = np.array(np.where(bar == 1))

    if bar_pixels.shape[1] == 0:
        return gray_Image.copy()

    first_bar_pixel = int(bar_pixels[:, 0][0])
    last_bar_pixel = int(bar_pixels[:, -1][0])

    if first_bar_pixel == 0 or last_bar_pixel == 0:
        return gray_Image.copy()

    x2 = int(bar_pos - CROP_RIGHT_MARGIN)

    if x2 <= left_bound:
        return gray_Image.copy()

    crop_img = gray_Image[
        first_bar_pixel:last_bar_pixel,
        left_bound:x2
    ].copy()

    if crop_img.size == 0 or crop_img.shape[0] < MIN_CROP_SIZE or crop_img.shape[1] < MIN_CROP_SIZE:
        return gray_Image.copy()

    return crop_img


def pixel_converter(orig_Image):
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

        _, threshold = cv2.threshold(gray_Image, BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        black = np.zeros_like(img_bin)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < PIXEL_CONTOUR_AREA:
                approx = cv2.approxPolyDP(
                    cnt,
                    0.00001 * cv2.arcLength(cnt, True),
                    True
                )

                if len(approx) == 4:
                    cv2.drawContours(black, [approx], 0, 255, -1)

                if len(approx) == 2:
                    cv2.drawContours(black, [approx], 0, 255, -1)

        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HORIZONTAL_KERNEL_SIZE)

        img_temp2 = cv2.erode(black, hori_kernel, iterations=MORPH_ITERATIONS)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=MORPH_ITERATIONS)

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


def reduce_color_palette(image, nr_of_colors):
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


def binarize(Image, tresh):
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


def identify_poly(X_, Y_, order):
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


def fit_points(pts, poly_line, deviation):
    point = []

    for p in pts:
        x_hat = poly_line(p.y)

        if abs(x_hat - p.x) < deviation:
            point.append(p)

    return point


def fit_points_with_clamp(pts, poly_line, deviation, X, Y):
    point = []
    X = []
    Y = []
    max_dev = MAX_FIT_DEVIATION

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


def compute_distance(p1, p2):
    return math.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2))


def remove_outliers(points, outlierConstant):
    if len(points) < 2:
        return np.zeros((0, 0))

    matrx = np.zeros(shape=(len(points), len(points)))

    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                dist = int(compute_distance(points[i], points[j]))

                if dist <= outlierConstant:
                    matrx[i][j] = 1

    return matrx


def extract_contour(Image, points):
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


def extract_connected_components(distances, points):
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


def connected_contour(ROI):
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


def find_all_contours_from_mask(mask):
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


def component_stats(component):
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


def is_pleura_like_component(component, principal_poly=None, image_shape=None, mode="normal"):
    stats = component_stats(component)

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


def is_principal_component_plausible(component, image_shape=None):
    stats = component_stats(component)

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


def compute_mask_metrics(mask, image_shape):
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


def score_pleura_result(result, image_shape, mode):
    mask = result["display_mask"]
    m = compute_mask_metrics(mask, image_shape)

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


    score += 4.0 * min(1.0, coverage / 0.55)
    score += 2.0 * min(1.0, width_frac / 0.65)
    score += 1.2 * min(1.0, continuity)


    if 0.10 <= y_center_frac <= 0.85:
        score += 1.0
    else:
        score -= 2.0
        reasons.append("bad_vertical_position")


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


    if mode == "surplus":
        score -= 3.5 * area_frac
        score -= 1.8 * height_frac
    else:
        score -= 1.5 * area_frac


    if mode == "pathologic":
        score -= 0.08 * max(0, components_count - 1)
    elif mode == "surplus":
        score -= 0.35 * max(0, components_count - 1)
    else:
        score -= 0.20 * max(0, components_count - 1)


    if mode == "pathologic":
        score -= 0.30 * min(3.0, thickness_var_norm)
    else:
        score -= 0.85 * min(3.0, thickness_var_norm)


    if mode == "pathologic":
        max_slope = 2.2
    else:
        max_slope = 1.4

    if slope > max_slope:
        score -= 1.5
        reasons.append("large_slope")


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


def run_automatic_pleura_detection(crop_Image_rgb, one_pixel=ONE_PIXEL_FALLBACK):
    candidates = []
    mode_errors = []

    for mode in AUTO_MODES:
        try:
            interpreted_Image = crop_Image_rgb.copy()

            result = extract_pleural_line(
                crop_Image_rgb,
                interpreted_Image,
                mode=mode,
                one_pixel=one_pixel
            )

            final_contours = result.get("final_contours", [])

            if final_contours is None or len(final_contours) == 0:
                raise ValueError("Nu s-a extras niciun contur final in modul " + mode)

            score, metrics = score_pleura_result(
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


def identify_principal_contour(img, deviation=DEFAULT_DEVIATION):
    points = []

    extract_contour(img, points)

    if len(points) < 2:
        raise ValueError("[identify_principal_contour] Prea putine puncte candidate.")

    distances = remove_outliers(points, deviation)

    pts, PX_, PY_ = extract_connected_components(distances, points)

    if len(PX_) < 2:
        raise ValueError("[identify_principal_contour] Componenta principala prea mica.")

    poly_line, PX_fit = identify_poly(PX_, PY_, 3)

    pleural_underline = fit_points(pts, poly_line, deviation)

    if len(pleural_underline) < 2:
        pleural_underline = pts

    if len(pleural_underline) < 2:
        raise ValueError("[identify_principal_contour] Pleura principala invalida.")

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
        raise ValueError("[identify_principal_contour] ROI principala invalida.")

    ROI = img[Xmin:Xmax, Ymin:Ymax]

    contours = connected_contour(ROI)

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

    pleura, PX_, PY_ = fit_points_with_clamp(pts, poly_line, DEFAULT_DEVIATION, PX, PY)

    if len(pleura) < 2:
        pleura = pts
        PX_ = [p.x for p in pleura]
        PY_ = [p.y for p in pleura]

    poly_line2, PX_fit2 = identify_poly(PX_, PY_, 1)

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


def identify_secondary_contour(lateral_poly, img, minX, minY):
    points = []

    extract_contour(img, points)

    if len(points) < 2:
        return False, False, False, False, 1

    distances = remove_outliers(points, DEFAULT_DEVIATION)

    pts, PX_, PY_ = extract_connected_components(distances, points)

    if len(PX_) < 2:
        return False, False, False, False, 1

    poly_line, PX_fit = identify_poly(PX_, PY_, 3)

    deviation = SECONDARY_INITIAL_DEVIATION
    pleural_underline = fit_points(pts, poly_line, deviation)

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

    contours = connected_contour(ROI)

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

    deviation = SECONDARY_LATERAL_DEVIATION
    pleura, PX_, PY_ = fit_points_with_clamp(pts, lateral_poly, deviation, PX, PY)

    if len(pleura) < 2:
        return False, False, False, False, 1

    ps = []

    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)

    return ctr, PY_, PX_, pleura, 0


def get_component_bounds(component):
    stats = component_stats(component)

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


def get_endpoint(component, side):
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


def detect_interruption_candidates(components, image_shape, one_pixel, mode="normal"):
    h, w = image_shape[:2]

    if len(components) < 2:
        return []

    valid_components = []

    for comp in components:
        bounds = get_component_bounds(comp)

        if bounds is not None:
            valid_components.append(comp)

    valid_components = sorted(
        valid_components,
        key=lambda c: get_component_bounds(c)["min_x"]
    )

    candidates = []

    min_gap_px = max(4, int(round(0.25 / max(one_pixel, 1e-6))))
    max_gap_px = max(14, int(round(2.50 / max(one_pixel, 1e-6))))

    max_vertical_diff = max(25, int(0.10 * h))

    for i in range(len(valid_components) - 1):
        left_comp = valid_components[i]
        right_comp = valid_components[i + 1]

        left_end = get_endpoint(left_comp, "right")
        right_start = get_endpoint(right_comp, "left")

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


def detect_nodule_candidates(components, image_shape, one_pixel, mode="normal"):
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


def build_component_mask(components, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for comp in components:
        if comp is False or comp is None:
            continue

        if len(comp) < 2:
            continue

        cv2.drawContours(mask, [comp], -1, 255, -1)

    return mask


def build_display_pleura_mask(components, image_shape, mode="normal"):
    h, w = image_shape[:2]
    mask = build_component_mask(components, image_shape)

    valid_components = []

    for comp in components:
        bounds = get_component_bounds(comp)

        if bounds is not None:
            valid_components.append(comp)

    if len(valid_components) <= 1:
        return mask

    valid_components = sorted(
        valid_components,
        key=lambda c: get_component_bounds(c)["min_x"]
    )

    if mode == "surplus":
        max_gap = max(20, int(0.035 * w))
        max_vertical_diff = max(18, int(0.035 * h))
        connection_thickness = 2

    elif mode == "pathologic":

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

        left_end = get_endpoint(left_comp, "right")
        right_start = get_endpoint(right_comp, "left")

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


def extract_pleural_line(orig_Image, interpreted_Image, mode="normal", one_pixel=ONE_PIXEL_FALLBACK):
    img = orig_Image.copy()

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    nr_of_colors = PLEURA_COLOR_COUNT
    img = reduce_color_palette(img, nr_of_colors)


    tresh = np.max(img) - PLEURA_THRESHOLD_OFFSET
    img = binarize(img, tresh)
    img = (1 - img).astype(np.uint8)

    left_contour_components = []
    right_contour_components = []
    accepted_components = []

    PrincipalComponent, PYS, poly_line2, points, retry = identify_principal_contour(img, DEFAULT_DEVIATION)

    principal_ok = is_principal_component_plausible(
        PrincipalComponent,
        image_shape=img.shape
    )

    if not principal_ok:
        pass

    accepted_components.append(PrincipalComponent)

    cv2.drawContours(
        interpreted_Image,
        [PrincipalComponent],
        0,
        (0, 255, 0),
        1
    )


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
        left_poly_line, _ = identify_poly(Ex, Ey, 3)

        newROI = img[minX:maxX, 0:Ymin]

        isEmpty = 0
        last_Y = Ymin + 5

        while Ymin < last_Y and newROI.shape[1] > 20 and isEmpty == 0:
            if len(newROI) > 0:
                NextComponent, PY_, PX_, points, isEmpty = identify_secondary_contour(
                    left_poly_line,
                    newROI,
                    minX,
                    0
                )
            else:
                break

            if isEmpty != 1:
                is_good = is_pleura_like_component(
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
        right_poly_line, _ = identify_poly(Ex, Ey, 3)

        newROI = img[minX:maxX, Ymax:img.shape[1] - 1]

        last_Y = Ymax - 5
        isEmpty = 0

        while Ymax > last_Y and newROI.shape[1] > 20 and isEmpty == 0:
            if len(newROI) > 0:
                NextComponent, PY_, PX_, points, isEmpty = identify_secondary_contour(
                    right_poly_line,
                    newROI,
                    minX,
                    Ymax
                )
            else:
                break

            if isEmpty != 1:
                is_good = is_pleura_like_component(
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

    component_mask = build_component_mask(
        accepted_components,
        interpreted_Image.shape[:2]
    )

    display_mask = build_display_pleura_mask(
        accepted_components,
        interpreted_Image.shape[:2],
        mode=mode
    )


    final_contours = find_all_contours_from_mask(display_mask)

    if len(final_contours) == 0:
        final_contours = accepted_components

    interruptions = detect_interruption_candidates(
        accepted_components,
        interpreted_Image.shape[:2],
        one_pixel,
        mode=mode
    )

    nodules = detect_nodule_candidates(
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


def load_image_rgb(img_path):
    image = io.imread(img_path)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image


def prepare_crop_rgb(image):
    crop = crop_border(image)

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    return crop, cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)


def get_one_pixel(image):
    try:
        return pixel_converter(image)
    except Exception:
        return ONE_PIXEL_FALLBACK


def build_auto_mode_row(img_name, mode, score, metrics):
    return {
        "image": img_name,
        "chosen_mode": mode,
        "score": score,
        "accepted": metrics.get("accepted", False),
        "suspect": metrics.get("suspect", True),
        "reason": metrics.get("reason", ""),
        "coverage": metrics.get("coverage", 0.0),
        "width_frac": metrics.get("width_frac", 0.0),
        "height_frac": metrics.get("height_frac", 0.0),
        "area_frac": metrics.get("area_frac", 0.0),
        "continuity": metrics.get("continuity", 0.0),
        "y_center_frac": metrics.get("y_center_frac", 0.0),
        "median_thickness": metrics.get("median_thickness", 0.0),
        "thickness_iqr": metrics.get("thickness_iqr", 0.0),
        "thickness_var_norm": metrics.get("thickness_var_norm", 0.0),
        "components_count": metrics.get("components_count", 0),
        "interruptions_count": metrics.get("interruptions_count", 0),
        "nodules_count": metrics.get("nodules_count", 0)
    }


def save_detection_outputs(idx, crop_rgb, result, output_dirs):
    final_contours = result["final_contours"]
    display_mask = result["display_mask"]
    interruptions = result.get("interruptions", [])
    nodules = result.get("nodules", [])
    prefix = format(idx, "02d")

    final_overlay = draw_final_overlay(crop_rgb, final_contours)
    diagnostic_overlay = draw_diagnostic_overlay(
        crop_rgb,
        final_contours,
        interruptions,
        nodules
    )

    save_rgb(
        os.path.join(output_dirs["contours"], prefix + "_detected_pleura.png"),
        final_overlay
    )

    save_rgb(
        os.path.join(output_dirs["diagnostic"], prefix + "_diagnostic.png"),
        diagnostic_overlay
    )

    cv2.imwrite(
        os.path.join(output_dirs["masks"], prefix + "_mask.png"),
        display_mask
    )

    return diagnostic_overlay


def write_error_file(error_dir, idx, img_name, error):
    error_path = os.path.join(error_dir, format(idx, "02d") + "_error.txt")

    with open(error_path, "w", encoding="utf-8") as f:
        f.write("Imagine: " + img_name + "\n")
        f.write("Eroare: " + str(error) + "\n\n")
        f.write(traceback.format_exc())


def write_summary_file(
    path,
    success_count,
    fail_count,
    suspect_count,
    auto_mode_report,
    failed_images,
    suspect_images,
    output_dirs,
    auto_mode_csv_path
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("BATCH PLEURA - MOD AUTOMAT\n")
        f.write("=" * 50 + "\n")
        f.write("Interval imagini: " + str(START_IDX) + "-" + str(END_IDX) + "\n")
        f.write("Reusite: " + str(success_count) + "\n")
        f.write("Esuate: " + str(fail_count) + "\n")
        f.write("Suspecte: " + str(suspect_count) + "\n\n")

        f.write("Output contururi: " + output_dirs["contours"] + "\n")
        f.write("Output diagnostic: " + output_dirs["diagnostic"] + "\n")
        f.write("Output masti: " + output_dirs["masks"] + "\n")
        f.write("Output suspecte: " + output_dirs["suspect"] + "\n")
        f.write("Output erori: " + output_dirs["errors"] + "\n")
        f.write("Raport CSV: " + auto_mode_csv_path + "\n\n")

        f.write("MOD ALES AUTOMAT PER IMAGINE:\n")
        for item in auto_mode_report:
            f.write(
                "  "
                + str(item["image"])
                + " | mode="
                + str(item["chosen_mode"])
                + " | score={:.4f}".format(item["score"])
                + " | suspect="
                + str(item["suspect"])
                + " | reason="
                + str(item["reason"])
                + "\n"
            )

        if failed_images:
            f.write("\nImagini esuate:\n")
            for name, err in failed_images:
                f.write("  " + name + ": " + err + "\n")

        if suspect_images:
            f.write("\nImagini suspecte:\n")
            for name, reason in suspect_images:
                f.write("  " + name + ": " + reason + "\n")


def main():
    output_dirs = {
        "contours": ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_CONTOURS")),
        "diagnostic": ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_DIAGNOSTIC")),
        "masks": ensure_dir(os.path.join(OUTPUT_ROOT, "ALL_MASKS")),
        "suspect": ensure_dir(os.path.join(OUTPUT_ROOT, "SUSPECT_RESULTS")),
        "errors": ensure_dir(os.path.join(OUTPUT_ROOT, "ERRORS"))
    }

    success_count = 0
    fail_count = 0
    suspect_count = 0

    failed_images = []
    suspect_images = []
    auto_mode_report = []

    total_images = END_IDX - START_IDX + 1

    for idx in range(START_IDX, END_IDX + 1):
        img_name = str(idx) + ".jpg"
        img_path = os.path.join(INPUT_DIR, img_name)
        current_image = idx - START_IDX + 1

        print("\n[" + str(current_image) + "/" + str(total_images) + "] Procesez: " + img_name)

        if not os.path.exists(img_path):
            fail_count += 1
            failed_images.append((img_name, "Fisier inexistent"))
            continue

        try:
            original_image = load_image_rgb(img_path)
            one_pixel = get_one_pixel(original_image)
            _, crop_rgb = prepare_crop_rgb(original_image)

            best_candidate, _, _ = run_automatic_pleura_detection(
                crop_rgb,
                one_pixel=one_pixel
            )

            mode = best_candidate["mode"]
            result = best_candidate["result"]
            auto_score = best_candidate["score"]
            auto_metrics = best_candidate["metrics"]
            final_contours = result.get("final_contours", [])

            if final_contours is None or len(final_contours) == 0:
                raise ValueError("Nu s-a extras niciun contur final.")

            diagnostic_overlay = save_detection_outputs(
                idx,
                crop_rgb,
                result,
                output_dirs
            )

            auto_mode_report.append(
                build_auto_mode_row(img_name, mode, auto_score, auto_metrics)
            )

            if auto_metrics.get("suspect", True) or not auto_metrics.get("accepted", False):
                suspect_count += 1
                reason = "Contur suspect automat: " + str(auto_metrics.get("reason", ""))
                suspect_images.append((img_name, reason))
                save_rgb(
                    os.path.join(output_dirs["suspect"], format(idx, "02d") + "_suspect.png"),
                    diagnostic_overlay
                )

            success_count += 1

        except Exception as e:
            fail_count += 1
            failed_images.append((img_name, str(e)))
            write_error_file(output_dirs["errors"], idx, img_name, e)

    summary_path = os.path.join(OUTPUT_ROOT, "summary.txt")
    auto_mode_csv_path = os.path.join(OUTPUT_ROOT, "auto_mode_report.csv")

    write_auto_mode_csv(auto_mode_csv_path, auto_mode_report)
    write_summary_file(
        summary_path,
        success_count,
        fail_count,
        suspect_count,
        auto_mode_report,
        failed_images,
        suspect_images,
        output_dirs,
        auto_mode_csv_path
    )


if __name__ == "__main__":
    main()
