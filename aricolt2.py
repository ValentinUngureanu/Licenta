import os
import csv
import shutil
from dataclasses import dataclass

import cv2
import numpy as np


INPUT_DIR = r"C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES"
OUTPUT_DIR = r"C:\Facultate\AN4\Licenta\Licenta-Cod\CROP_RESULTS"
FINAL_CONTOUR_ON_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "FINAL_CONTOUR_ON_ORIGINAL")
DEBUG_IMPORTANT_STEPS_DIR = os.path.join(OUTPUT_DIR, "DEBUG_IMPORTANT_STEPS")
IDENTIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "identification_report.csv")

RESET_OUTPUT_DIR_ON_RUN = True

START_IDX = 0
END_IDX = 61
SINGLE_IMAGE_IDX = 1
RUN_SINGLE_IMAGE = False

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

MAX_PIXEL_VALUE = 255
DEFAULT_BINARY_THRESHOLD = 128
BINARY_THRESHOLD = 110
MIN_BORDER_AREA_FRAC = 0.0007
SMALL_CONTOUR_AREA_FRAC = 0.00003
LEFT_MARGIN_SEARCH_FRAC = 0.07
DEFAULT_LEFT_BOUND_FRAC = 0.018
CROP_RIGHT_MARGIN_FRAC = 0.014
MIN_CROP_SIZE_FRAC = 0.10
HORIZONTAL_KERNEL_WIDTH_FRAC = 0.0015
MORPH_ITERATIONS = 3
SUSPECT_FULL_CROP_FRAC = 0.90

EXTRA_CROP_TOP_FRAC = 0.035
EXTRA_CROP_BOTTOM_FRAC = 0.035
EXTRA_CROP_LEFT_FRAC = 0.0
EXTRA_CROP_RIGHT_FRAC = 0.0
EXTRA_CROP_TOP_PX = 0
EXTRA_CROP_BOTTOM_PX = 0
EXTRA_CROP_LEFT_PX = 0
EXTRA_CROP_RIGHT_PX = 0

HORIZONTAL_CROP_TOP_IGNORE_FRAC = 0.08
HORIZONTAL_CROP_BOTTOM_IGNORE_FRAC = 0.08
HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC = 0.35
HORIZONTAL_MAX_FULL_WIDTH_FRAC = 0.96
HORIZONTAL_PAD_FRAC = 0.012
HORIZONTAL_SMOOTH_FRAC = 0.030
HORIZONTAL_THRESHOLD_PERCENTILES = [45, 50, 55, 60, 65, 70, 75]
HORIZONTAL_THRESHOLD_SCALE = 0.65

PALETTE_COLORS = 7

# Variante:
#   "percentile_linear" = metoda veche: normalizare percentila + impartire egala
#   "clahe_kmeans"      = metoda noua: contrast local + KMeans pe intensitati
PALETTE_METHOD = "clahe_kmeans"
PALETTE_PERCENTILE_LOW = 2
PALETTE_PERCENTILE_HIGH = 99
PALETTE_CLAHE_CLIP_LIMIT = 2.0
PALETTE_CLAHE_TILE_SIZE = 8
PALETTE_KMEANS_ATTEMPTS = 3
PALETTE_KMEANS_MAX_ITER = 35
PALETTE_VALID_LOW_PERCENTILE = 1
PALETTE_VALID_HIGH_PERCENTILE = 99.7
BINARY_KEEP_TOP_LEVELS = 1
OUTLIER_DISTANCE = 50
POLY_DEGREE = 3
POLY_DEVIATION = 20
POLY_MIN_POINTS = 10
COMPONENT_MIN_POINTS = 8
COMPONENT_MIN_WIDTH_FRAC = 0.035
COMPONENT_TOO_LOW_FRAC = 0.70
COMPONENT_TOO_HIGH_FRAC = 0.06
BAND_BIN_HEIGHT_FRAC = 0.018
BAND_KEEP_HALF_HEIGHT_FRAC = 0.045
BAND_MIN_POINTS = 8
ROI_PAD_X = 10
ROI_PAD_Y_TOP = 10
ROI_PAD_Y_BOTTOM = 10
PRINCIPAL_ROI_MIN_HEIGHT_FRAC = 0.025
PRINCIPAL_ROI_MIN_WIDTH_FRAC = 0.050
REMOVE_BELOW_TRAVELERS_OFFSET = 10
PRINCIPAL_COMPONENT_MIN_AREA_FRAC = 0.00001
PRINCIPAL_COMPONENT_DILATE_KERNEL = 3
PRINCIPAL_CONTOUR_DEVIATION = 50
FINAL_POLY_DEGREE = 1
SECONDARY_LEFT_EXTRA_COLS = 10
SECONDARY_RIGHT_EXTRA_COLS = 10
SECONDARY_TOP_MARGIN_LEFT = 40
SECONDARY_TOP_MARGIN_RIGHT = 30
SECONDARY_CONTOUR_DEVIATION = 100
SECONDARY_POLY_BAND_ABOVE_PX = 25
SECONDARY_POLY_BAND_BELOW_PX = 25
SECONDARY_KEEP_LARGEST_IN_BAND = True
SECONDARY_MIN_POINTS = 6
MERGE_MIN_COMPONENT_AREA = 30
MERGE_CONNECTION_THICKNESS = 3
MERGE_ENDPOINT_SAMPLE = 25
DEBUG_CONSOLE = True
DEBUG_SAVE_IMPORTANT_STEPS = True
WRITE_IDENTIFICATION_REPORT = True
PLEURA_THICKNESS_ABOVE_PX = 18
PLEURA_THICKNESS_BELOW_PX = 22
PLEURA_LIMIT_KEEP_LARGEST = True
FINAL_CONTOUR_THICKNESS = 1


@dataclass
class CropBox:
    top: int
    left: int
    bottom: int
    right: int
    valid: bool = True

    @property
    def width(self):
        return max(0, self.right - self.left)

    @property
    def height(self):
        return max(0, self.bottom - self.top)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    dist: float = 0.0


def reset_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

    return path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def find_image_path(index):
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(INPUT_DIR, str(index) + ext)
        if os.path.exists(path):
            return path
    return None


def read_image_bgr(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Nu pot citi imaginea: " + path)
    return image


def to_gray(image_bgr):
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def clamp_box(box, shape):
    h, w = shape[:2]
    top = int(max(0, min(box.top, h - 1)))
    bottom = int(max(top + 1, min(box.bottom, h)))
    left = int(max(0, min(box.left, w - 1)))
    right = int(max(left + 1, min(box.right, w)))
    return CropBox(top, left, bottom, right, box.valid)


def is_box_valid(box, shape):
    h, w = shape[:2]
    min_size = int(max(30, MIN_CROP_SIZE_FRAC * min(h, w)))
    return box.width >= min_size and box.height >= min_size


def crop_is_suspect_full(box, shape):
    h, w = shape[:2]
    return box.width >= SUSPECT_FULL_CROP_FRAC * w and box.height >= SUSPECT_FULL_CROP_FRAC * h


def get_contours(binary):
    found = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(found) == 2:
        contours, _ = found
    else:
        _, contours, _ = found
    return contours


def largest_true_segment(flags, min_len):
    best_start = None
    best_end = None
    best_len = 0
    start = None

    for i, value in enumerate(flags):
        if value and start is None:
            start = i
        elif not value and start is not None:
            end = i
            length = end - start
            if length > best_len and length >= min_len:
                best_len = length
                best_start = start
                best_end = end
            start = None

    if start is not None:
        end = len(flags)
        length = end - start
        if length > best_len and length >= min_len:
            best_start = start
            best_end = end

    if best_start is None:
        return None

    return best_start, best_end


def estimate_crop_box_from_bar(gray):
    h, w = gray.shape[:2]
    image_area = h * w
    min_border_area = max(150, int(MIN_BORDER_AREA_FRAC * image_area))
    small_contour_area = max(8, int(SMALL_CONTOUR_AREA_FRAC * image_area))
    left_margin_limit = max(30, int(LEFT_MARGIN_SEARCH_FRAC * w))
    default_left_bound = max(5, int(DEFAULT_LEFT_BOUND_FRAC * w))
    crop_right_margin = max(8, int(CROP_RIGHT_MARGIN_FRAC * w))
    kernel_w = max(2, int(HORIZONTAL_KERNEL_WIDTH_FRAC * w))

    _, img_bin = cv2.threshold(gray, DEFAULT_BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)
    _, threshold = cv2.threshold(gray, BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)
    contours = get_contours(threshold)

    large_mask = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_border_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) in [2, 4]:
            cv2.drawContours(large_mask, [approx], 0, 255, -1)

    white_pixels = np.array(np.where(large_mask == 255))

    if white_pixels.shape[1] == 0:
        left_bound = default_left_bound
    else:
        left_candidates = white_pixels[1, white_pixels[1] < left_margin_limit]
        if len(left_candidates) == 0:
            left_bound = default_left_bound
        else:
            left_bound = int(np.percentile(left_candidates, 95))

    small_mask = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= small_contour_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.00001 * peri, True)
        if len(approx) in [2, 4]:
            cv2.drawContours(small_mask, [approx], 0, 255, -1)

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    temp = cv2.erode(small_mask, hori_kernel, iterations=MORPH_ITERATIONS)
    horizontal_lines = cv2.dilate(temp, hori_kernel, iterations=MORPH_ITERATIONS)
    columns = np.count_nonzero(horizontal_lines, axis=0)

    if np.max(columns) == 0:
        return CropBox(0, 0, h, w, False)

    bar_pos = int(np.argmax(columns))
    bar = horizontal_lines[:, bar_pos] // 255
    bar_pixels = np.where(bar == 1)[0]

    if len(bar_pixels) == 0:
        return CropBox(0, 0, h, w, False)

    box = CropBox(
        top=int(bar_pixels[0]),
        left=int(left_bound),
        bottom=int(bar_pixels[-1]),
        right=int(bar_pos - crop_right_margin),
        valid=True
    )
    box = clamp_box(box, gray.shape)

    if not is_box_valid(box, gray.shape) or crop_is_suspect_full(box, gray.shape):
        box.valid = False

    return box


def estimate_crop_box_from_activity(gray):
    h, w = gray.shape[:2]
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    non_black = (gray_blur > 12) & (gray_blur < 245)

    row_density = np.count_nonzero(non_black, axis=1) / max(w, 1)
    col_density = np.count_nonzero(non_black, axis=0) / max(h, 1)

    row_smooth = cv2.blur(row_density.astype(np.float32).reshape(-1, 1), (1, 21)).reshape(-1)
    col_smooth = cv2.blur(col_density.astype(np.float32).reshape(1, -1), (31, 1)).reshape(-1)

    row_thr = max(0.025, float(np.percentile(row_smooth, 65)) * 0.45)
    col_thr = max(0.020, float(np.percentile(col_smooth, 65)) * 0.45)

    row_segment = largest_true_segment(row_smooth > row_thr, int(0.25 * h))
    col_segment = largest_true_segment(col_smooth > col_thr, int(0.35 * w))

    if row_segment is None or col_segment is None:
        return CropBox(0, 0, h, w, False)

    top, bottom = row_segment
    left, right = col_segment
    pad_y = int(0.015 * h)
    pad_x = int(0.015 * w)

    box = CropBox(
        top=max(0, top - pad_y),
        left=max(0, left - pad_x),
        bottom=min(h, bottom + pad_y),
        right=min(w, right + pad_x),
        valid=True
    )
    box = clamp_box(box, gray.shape)

    if not is_box_valid(box, gray.shape) or crop_is_suspect_full(box, gray.shape):
        box.valid = False

    return box


def choose_crop_box(gray):
    bar_box = estimate_crop_box_from_bar(gray)
    activity_box = estimate_crop_box_from_activity(gray)
    candidates = []

    if bar_box.valid:
        candidates.append(bar_box)

    if activity_box.valid:
        candidates.append(activity_box)

    if len(candidates) == 0:
        return CropBox(0, 0, gray.shape[0], gray.shape[1], False)

    def score_box(box):
        h, w = gray.shape[:2]
        area_frac = (box.width * box.height) / max(h * w, 1)
        center_y = (box.top + box.bottom) / 2.0 / max(h, 1)
        center_x = (box.left + box.right) / 2.0 / max(w, 1)
        score = 0.0
        score += 2.0 * min(1.0, box.width / max(0.55 * w, 1))
        score += 2.0 * min(1.0, box.height / max(0.45 * h, 1))
        score -= 1.2 * abs(area_frac - 0.55)
        score -= 0.6 * abs(center_x - 0.50)
        if 0.25 <= center_y <= 0.70:
            score += 0.5
        return score

    return max(candidates, key=score_box)


def estimate_left_right_from_vertical_crop(gray_vertical):
    h, w = gray_vertical.shape[:2]

    if h < 30 or w < 30:
        return CropBox(0, 0, h, w, False)

    y1 = int(round(HORIZONTAL_CROP_TOP_IGNORE_FRAC * h))
    y2 = int(round((1.0 - HORIZONTAL_CROP_BOTTOM_IGNORE_FRAC) * h))
    y1 = max(0, min(y1, h - 2))
    y2 = max(y1 + 1, min(y2, h))

    roi = gray_vertical[y1:y2, :].copy()

    if roi.size == 0:
        return CropBox(0, 0, h, w, False)

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    non_black = (blur > 10) & (blur < 250)
    non_black_density = np.count_nonzero(non_black, axis=0) / max(roi.shape[0], 1)

    edges = cv2.Canny(blur, 30, 90)
    edge_density = np.count_nonzero(edges > 0, axis=0) / max(roi.shape[0], 1)

    if np.max(edge_density) > 0:
        edge_density = edge_density / np.max(edge_density)

    signal = 0.80 * non_black_density + 0.20 * edge_density

    smooth_w = max(9, int(round(HORIZONTAL_SMOOTH_FRAC * w)))
    if smooth_w % 2 == 0:
        smooth_w += 1

    signal_smooth = cv2.blur(signal.astype(np.float32).reshape(1, -1), (smooth_w, 1)).reshape(-1)
    candidates = []
    min_len = max(20, int(round(HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC * w)))

    for perc in HORIZONTAL_THRESHOLD_PERCENTILES:
        base = float(np.percentile(signal_smooth, perc))
        thr = max(0.010, base * HORIZONTAL_THRESHOLD_SCALE)
        segment = largest_true_segment(signal_smooth > thr, min_len)

        if segment is None:
            continue

        left, right = segment
        pad = max(4, int(round(HORIZONTAL_PAD_FRAC * w)))
        left = max(0, left - pad)
        right = min(w, right + pad)
        width = max(1, right - left)
        width_frac = width / max(w, 1)
        center_frac = ((left + right) / 2.0) / max(w, 1)

        if width_frac < HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC:
            continue

        score = 0.0
        score += 2.5 * min(1.0, width_frac / 0.75)
        score -= 1.2 * abs(center_frac - 0.50)
        score -= 2.0 * max(0.0, width_frac - HORIZONTAL_MAX_FULL_WIDTH_FRAC)
        score += 0.01 * perc

        candidates.append((score, left, right))

    if len(candidates) == 0:
        thr = max(0.010, float(np.median(signal_smooth)))
        xs = np.where(signal_smooth > thr)[0]

        if len(xs) == 0:
            return CropBox(0, 0, h, w, False)

        pad = max(4, int(round(HORIZONTAL_PAD_FRAC * w)))
        left = max(0, int(np.min(xs)) - pad)
        right = min(w, int(np.max(xs)) + 1 + pad)
        return clamp_box(CropBox(0, left, h, right, True), gray_vertical.shape)

    _, left, right = max(candidates, key=lambda item: item[0])
    box = CropBox(0, left, h, right, True)
    box = clamp_box(box, gray_vertical.shape)

    if not is_box_valid(box, gray_vertical.shape):
        box.valid = False

    return box


def crop_border(image_bgr):
    gray = to_gray(image_bgr)
    h, w = gray.shape[:2]

    vertical_base = choose_crop_box(gray)
    vertical_box = CropBox(vertical_base.top, 0, vertical_base.bottom, w, vertical_base.valid)
    vertical_box = clamp_box(vertical_box, gray.shape)

    gray_vertical = gray[vertical_box.top:vertical_box.bottom, :].copy()
    horizontal_box = estimate_left_right_from_vertical_crop(gray_vertical)

    if horizontal_box.valid:
        final_box = CropBox(
            top=vertical_box.top,
            left=horizontal_box.left,
            bottom=vertical_box.bottom,
            right=horizontal_box.right,
            valid=True
        )
    else:
        final_box = CropBox(
            top=vertical_box.top,
            left=0,
            bottom=vertical_box.bottom,
            right=w,
            valid=vertical_box.valid
        )

    final_box = clamp_box(final_box, gray.shape)

    add_top = max(EXTRA_CROP_TOP_PX, int(round(EXTRA_CROP_TOP_FRAC * final_box.height)))
    add_bottom = max(EXTRA_CROP_BOTTOM_PX, int(round(EXTRA_CROP_BOTTOM_FRAC * final_box.height)))
    add_left = max(EXTRA_CROP_LEFT_PX, int(round(EXTRA_CROP_LEFT_FRAC * final_box.width)))
    add_right = max(EXTRA_CROP_RIGHT_PX, int(round(EXTRA_CROP_RIGHT_FRAC * final_box.width)))

    final_box = CropBox(
        top=final_box.top + add_top,
        left=final_box.left + add_left,
        bottom=final_box.bottom - add_bottom,
        right=final_box.right - add_right,
        valid=final_box.valid
    )
    final_box = clamp_box(final_box, gray.shape)

    crop = gray[final_box.top:final_box.bottom, final_box.left:final_box.right].copy()

    return crop, final_box


def reduce_palette_7_percentile_linear(gray, colors=7):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.uint8)

    low = np.percentile(gray, PALETTE_PERCENTILE_LOW)
    high = np.percentile(gray, PALETTE_PERCENTILE_HIGH)

    if high <= low:
        return gray.copy()

    normalized = np.clip((gray.astype(np.float32) - low) / (high - low), 0, 1)
    quantized = np.floor(normalized * colors)

    quantized[quantized >= colors] = colors - 1

    result = (quantized / (colors - 1) * 255).astype(np.uint8)

    return result


def reduce_palette_7_clahe_kmeans(gray, colors=7):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.uint8)

    tile_size = max(2, int(PALETTE_CLAHE_TILE_SIZE))
    clahe = cv2.createCLAHE(
        clipLimit=float(PALETTE_CLAHE_CLIP_LIMIT),
        tileGridSize=(tile_size, tile_size)
    )

    enhanced = clahe.apply(gray)

    # O netezire mică reduce sclipirile izolate fără să distrugă liniile mari.
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    valid_low = np.percentile(enhanced, PALETTE_VALID_LOW_PERCENTILE)
    valid_high = np.percentile(enhanced, PALETTE_VALID_HIGH_PERCENTILE)

    valid_mask = (enhanced >= valid_low) & (enhanced <= valid_high)

    samples = enhanced[valid_mask].reshape(-1, 1).astype(np.float32)

    if samples.shape[0] < colors * 20:
        return reduce_palette_7_percentile_linear(gray, colors=colors)

    cv2.setRNGSeed(12345)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(PALETTE_KMEANS_MAX_ITER),
        0.4
    )

    _, labels, centers = cv2.kmeans(
        samples,
        int(colors),
        None,
        criteria,
        int(PALETTE_KMEANS_ATTEMPTS),
        cv2.KMEANS_PP_CENTERS
    )

    centers = centers.reshape(-1).astype(np.float32)
    order = np.argsort(centers)
    centers_sorted = centers[order]

    # Nuanțe finale fixe: 0, 42, 85, ..., 255.
    # Asta păstrează compatibilitatea cu binarize_palette_7.
    output_levels = np.linspace(0, 255, int(colors)).astype(np.uint8)

    flat = enhanced.reshape(-1).astype(np.float32)
    distances = np.abs(flat[:, None] - centers_sorted[None, :])
    nearest = np.argmin(distances, axis=1)

    result = output_levels[nearest].reshape(enhanced.shape).astype(np.uint8)

    # Pixeli foarte negri rămân pe nivelul minim.
    dark_cut = max(3, np.percentile(gray, 0.5))
    result[gray <= dark_cut] = 0

    return result


def reduce_palette_7(gray, colors=7):
    if PALETTE_METHOD == "clahe_kmeans":
        return reduce_palette_7_clahe_kmeans(gray, colors=colors)

    return reduce_palette_7_percentile_linear(gray, colors=colors)


def binarize_palette_7(palette_gray, keep_top_levels=BINARY_KEEP_TOP_LEVELS):
    if palette_gray.ndim == 3:
        palette_gray = cv2.cvtColor(palette_gray, cv2.COLOR_BGR2GRAY)

    palette_gray = palette_gray.astype(np.uint8)
    values = np.sort(np.unique(palette_gray))

    if len(values) < 2:
        return np.zeros_like(palette_gray, dtype=np.uint8), 0

    keep_top_levels = max(1, min(keep_top_levels, len(values)))
    threshold = int(values[-keep_top_levels])

    binary = (palette_gray >= threshold).astype(np.uint8) * 255

    return binary, threshold


def ExtractContour(image, points):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = (image > 0).astype(np.uint8)

    for y in range(0, binary.shape[1] - 1, 1):
        for x in range(binary.shape[0] - 1, 0, -1):
            if binary[x, y] == 1:
                p = Point()
                p.x = x
                p.y = y
                points.append(p)
                break


def draw_traveler_points(crop_gray, points):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    for p in points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 1, (255, 0, 0), -1)

    return result


def ComputeDistance(p1, p2):
    return float(np.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2)))


def removeOutliers(points, outlierConstant):
    n = len(points)

    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    matrix = np.zeros((n, n), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            distance = ComputeDistance(points[i], points[j])

            if distance <= outlierConstant:
                matrix[i, j] = 1

    return matrix


def ExtractConnectedComponents(distances, points):
    if len(points) == 0 or distances.size == 0:
        return [], [], []

    n = len(points)
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while len(stack) > 0:
            current = stack.pop()
            component.append(current)

            neighbors = np.where(distances[current] > 0)[0]

            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        components.append(component)

    if len(components) == 0:
        return [], [], []

    main_indices = max(components, key=len)

    pts = []
    X_ = []
    Y_ = []

    for idx in main_indices:
        p = points[idx]
        pts.append(p)
        X_.append(p.x)
        Y_.append(p.y)

    return pts, X_, Y_


def draw_main_traveler_component(crop_gray, all_points, main_points):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    for p in all_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 1, (80, 80, 80), -1)

    for p in main_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 2, (0, 255, 0), -1)

    return result


def extract_all_traveler_components(distances, points):
    if len(points) == 0 or distances.size == 0:
        return []

    n = len(points)
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component_indices = []

        while len(stack) > 0:
            current = stack.pop()
            component_indices.append(current)

            neighbors = np.where(distances[current] > 0)[0]

            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        component_points = [points[idx] for idx in component_indices]
        components.append(component_points)

    return components


def score_traveler_component(component, image_shape):
    h, w = image_shape[:2]

    if len(component) == 0:
        return -1e9, {}

    rows = np.array([p.x for p in component], dtype=np.float32)
    cols = np.array([p.y for p in component], dtype=np.float32)

    row_min = float(np.min(rows))
    row_max = float(np.max(rows))
    col_min = float(np.min(cols))
    col_max = float(np.max(cols))

    row_span = max(1.0, row_max - row_min + 1.0)
    col_span = max(1.0, col_max - col_min + 1.0)

    width_frac = col_span / max(float(w), 1.0)
    height_frac = row_span / max(float(h), 1.0)
    median_row_frac = float(np.median(rows)) / max(float(h), 1.0)
    aspect = col_span / row_span

    if len(component) >= 3 and len(np.unique(cols)) >= 3:
        try:
            slope = float(np.polyfit(cols, rows, 1)[0])
        except Exception:
            slope = 999.0
    else:
        slope = 999.0

    score = 0.0

    score += 4.0 * min(width_frac / 0.45, 1.0)
    score += 2.0 * min(len(component) / 140.0, 1.0)
    score += 1.8 * min(aspect / 12.0, 1.0)

    score -= 2.8 * min(abs(slope), 3.0)
    score -= 1.2 * min(height_frac / 0.25, 1.0)

    score -= 5.0 * max(0.0, median_row_frac - COMPONENT_TOO_LOW_FRAC)
    score -= 3.0 * max(0.0, COMPONENT_TOO_HIGH_FRAC - median_row_frac)

    if COMPONENT_TOO_HIGH_FRAC <= median_row_frac <= COMPONENT_TOO_LOW_FRAC:
        score += 1.0

    if len(component) < COMPONENT_MIN_POINTS:
        score -= 4.0

    if width_frac < COMPONENT_MIN_WIDTH_FRAC:
        score -= 4.0

    info = {
        "score": float(score),
        "count": int(len(component)),
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
        "width_frac": float(width_frac),
        "height_frac": float(height_frac),
        "median_row_frac": float(median_row_frac),
        "aspect": float(aspect),
        "slope": float(slope)
    }

    return score, info


def ExtractBestPleuralTravelerComponent(distances, points, image_shape):
    components = extract_all_traveler_components(distances, points)

    if len(components) == 0:
        return [], [], [], []

    scored_components = []

    for idx, component in enumerate(components):
        score, info = score_traveler_component(component, image_shape)
        info["component_index"] = int(idx)
        scored_components.append({
            "points": component,
            "score": float(score),
            "info": info
        })

    selected = max(scored_components, key=lambda item: item["score"])
    selected_points = selected["points"]

    X_ = [p.x for p in selected_points]
    Y_ = [p.y for p in selected_points]

    return selected_points, X_, Y_, scored_components


def draw_all_traveler_components(crop_gray, scored_components):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (180, 180, 0),
        (180, 0, 180),
        (0, 180, 180),
        (120, 120, 255)
    ]

    scored_components_sorted = sorted(
        scored_components,
        key=lambda item: item["score"],
        reverse=True
    )

    for rank, item in enumerate(scored_components_sorted):
        color = colors[rank % len(colors)]
        radius = 2 if rank == 0 else 1

        for p in item["points"]:
            row = int(round(p.x))
            col = int(round(p.y))

            if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
                cv2.circle(result, (col, row), radius, color, -1)

    return result


def draw_selected_pleural_travelers(crop_gray, all_points, selected_points):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    for p in all_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 1, (80, 80, 80), -1)

    for p in selected_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 2, (0, 255, 0), -1)

    return result


def filter_travelers_by_main_horizontal_band(points, image_shape):
    if points is None or len(points) == 0:
        return [], None

    h, w = image_shape[:2]

    rows = np.array([p.x for p in points], dtype=np.float32)

    bin_height = max(3, int(round(BAND_BIN_HEIGHT_FRAC * h)))
    half_height = max(5, int(round(BAND_KEEP_HALF_HEIGHT_FRAC * h)))

    bins = np.arange(0, h + bin_height, bin_height)

    if len(bins) < 2:
        return points, None

    hist, edges = np.histogram(rows, bins=bins)

    if len(hist) == 0 or np.max(hist) == 0:
        return points, None

    best_bin_index = int(np.argmax(hist))
    band_center = int(round((edges[best_bin_index] + edges[best_bin_index + 1]) / 2.0))

    filtered = []

    for p in points:
        if abs(float(p.x) - float(band_center)) <= half_height:
            filtered.append(p)

    if len(filtered) < BAND_MIN_POINTS:
        return points, {
            "band_center": band_center,
            "half_height": half_height,
            "used_fallback": True
        }

    return filtered, {
        "band_center": band_center,
        "half_height": half_height,
        "used_fallback": False
    }


def draw_band_filtered_travelers(crop_gray, selected_points, band_points, band_info):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    for p in selected_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 1, (80, 80, 80), -1)

    for p in band_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 2, (0, 255, 0), -1)

    if band_info is not None:
        h, w = result.shape[:2]
        center = int(band_info["band_center"])
        half_height = int(band_info["half_height"])

        y1 = max(0, center - half_height)
        y2 = min(h - 1, center + half_height)

        cv2.line(result, (0, center), (w - 1, center), (0, 255, 255), 1)
        cv2.line(result, (0, y1), (w - 1, y1), (0, 165, 255), 1)
        cv2.line(result, (0, y2), (w - 1, y2), (0, 165, 255), 1)

    return result


def IdnetifyPoly(X_, Y_, order):
    if len(X_) < order + 1 or len(Y_) < order + 1:
        return None, []

    poly_line = np.poly1d(np.polyfit(Y_, X_, order))
    fitted_X = poly_line(Y_)

    return poly_line, fitted_X


def Fit(pts, poly_line, deviation):
    if poly_line is None:
        return []

    filtered_points = []

    for p in pts:
        x_hat = poly_line(p.y)

        if abs(x_hat - p.x) < deviation:
            filtered_points.append(p)

    return filtered_points


def draw_poly_filtered_travelers(crop_gray, main_points, filtered_points, poly_line):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    for p in main_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 1, (80, 80, 80), -1)

    for p in filtered_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 2, (0, 255, 0), -1)

    if poly_line is not None:
        h, w = result.shape[:2]
        line_points = []

        for col in range(w):
            row = int(round(poly_line(col)))

            if 0 <= row < h:
                line_points.append([col, row])

        if len(line_points) >= 2:
            line_points = np.array(line_points, dtype=np.int32)
            cv2.polylines(result, [line_points], False, (0, 165, 255), 1, cv2.LINE_AA)

    return result


def is_principal_roi_valid(box, shape):
    h, w = shape[:2]

    if box is None:
        return False

    min_height = max(18, int(round(PRINCIPAL_ROI_MIN_HEIGHT_FRAC * h)))
    min_width = max(35, int(round(PRINCIPAL_ROI_MIN_WIDTH_FRAC * w)))

    return box.height >= min_height and box.width >= min_width

def build_principal_roi_from_travelers(binary_mask, filtered_points):
    if binary_mask.ndim == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    h, w = binary_mask.shape[:2]
    working = binary_mask.copy()

    if filtered_points is None or len(filtered_points) == 0:
        return working, CropBox(0, 0, h, w, False), working.copy()

    for p in filtered_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < h and 0 <= col < w:
            cut_start = min(h, row + REMOVE_BELOW_TRAVELERS_OFFSET)
            working[cut_start:h, col:col + 1] = 0

    rows = np.array([p.x for p in filtered_points], dtype=np.float32)
    cols = np.array([p.y for p in filtered_points], dtype=np.float32)

    top = int(np.floor(np.min(rows))) - ROI_PAD_Y_TOP
    bottom = int(np.ceil(np.max(rows))) + ROI_PAD_Y_BOTTOM + 1
    left = int(np.floor(np.min(cols))) - ROI_PAD_X
    right = int(np.ceil(np.max(cols))) + ROI_PAD_X + 1

    roi_box = CropBox(
        top=max(0, top),
        left=max(0, left),
        bottom=min(h, bottom),
        right=min(w, right),
        valid=True
    )

    roi_box = clamp_box(roi_box, binary_mask.shape)

    if not is_principal_roi_valid(roi_box, binary_mask.shape):
        roi_box.valid = False

    roi = working[roi_box.top:roi_box.bottom, roi_box.left:roi_box.right].copy()

    return working, roi_box, roi


def draw_principal_roi_overlay(crop_gray, roi_box, filtered_points):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if roi_box is not None and roi_box.valid:
        cv2.rectangle(
            result,
            (roi_box.left, roi_box.top),
            (roi_box.right - 1, roi_box.bottom - 1),
            (0, 255, 255),
            2
        )

    for p in filtered_points:
        row = int(round(p.x))
        col = int(round(p.y))

        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            cv2.circle(result, (col, row), 2, (0, 255, 0), -1)

    return result


def extract_largest_component_contour_from_roi(principal_roi, roi_box, full_shape):
    full_mask = np.zeros(full_shape[:2], dtype=np.uint8)
    full_contour_mask = np.zeros(full_shape[:2], dtype=np.uint8)

    if principal_roi is None or principal_roi.size == 0:
        return full_mask, full_contour_mask, None

    if roi_box is None or not roi_box.valid:
        return full_mask, full_contour_mask, None

    if principal_roi.ndim == 3:
        principal_roi = cv2.cvtColor(principal_roi, cv2.COLOR_BGR2GRAY)

    binary = (principal_roi > 0).astype(np.uint8)

    if np.count_nonzero(binary) == 0:
        return full_mask, full_contour_mask, None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return full_mask, full_contour_mask, None

    min_area = max(
        3,
        int(round(PRINCIPAL_COMPONENT_MIN_AREA_FRAC * full_shape[0] * full_shape[1]))
    )

    best_label = None
    best_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return full_mask, full_contour_mask, None

    component_roi = np.zeros_like(binary, dtype=np.uint8)
    component_roi[labels == best_label] = 255

    kernel_size = max(1, int(PRINCIPAL_COMPONENT_DILATE_KERNEL))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    component_roi = cv2.dilate(component_roi, kernel, iterations=1)

    contours = get_contours(component_roi)

    if len(contours) == 0:
        return full_mask, full_contour_mask, None

    largest_contour = max(contours, key=cv2.contourArea)

    contour_full = largest_contour.copy()
    contour_full[:, 0, 0] += roi_box.left
    contour_full[:, 0, 1] += roi_box.top

    full_mask[
        roi_box.top:roi_box.bottom,
        roi_box.left:roi_box.right
    ] = component_roi[:roi_box.height, :roi_box.width]

    cv2.drawContours(full_contour_mask, [contour_full], -1, 255, 1)

    return full_mask, full_contour_mask, contour_full


def draw_principal_component_overlay(crop_gray, contour_mask):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if contour_mask is not None:
        result[contour_mask > 0] = (0, 255, 0)

    return result


def contour_to_points(contour):
    points = []

    if contour is None:
        return points

    for item in contour:
        col = float(item[0][0])
        row = float(item[0][1])

        p = Point()
        p.x = row
        p.y = col
        points.append(p)

    return points


def Fit2(pts, poly_line, deviation):
    if poly_line is None:
        return [], [], []

    filtered_points = []
    X_ = []
    Y_ = []
    max_deviation = 300

    for p in pts:
        x_hat = poly_line(p.y)

        if abs(x_hat - p.x) < deviation:
            filtered_points.append(p)
            X_.append(p.x)
            Y_.append(p.y)

        elif abs(x_hat - p.x) < max_deviation:
            corrected = Point()
            corrected.y = p.y

            if x_hat - p.x < 0:
                corrected.x = x_hat + deviation
            else:
                corrected.x = x_hat - deviation

            filtered_points.append(corrected)
            X_.append(corrected.x)
            Y_.append(corrected.y)

    return filtered_points, X_, Y_


def points_to_contour(points):
    if points is None or len(points) == 0:
        return None

    contour_points = []

    for p in points:
        col = int(round(p.y))
        row = int(round(p.x))
        contour_points.append([col, row])

    return np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))


def build_filtered_principal_component(contour, poly_line, shape):
    filtered_mask = np.zeros(shape[:2], dtype=np.uint8)
    filtered_contour_mask = np.zeros(shape[:2], dtype=np.uint8)

    contour_points = contour_to_points(contour)

    if len(contour_points) == 0:
        return filtered_mask, filtered_contour_mask, None, None, []

    filtered_points, X_, Y_ = Fit2(
        contour_points,
        poly_line,
        PRINCIPAL_CONTOUR_DEVIATION
    )

    filtered_contour = points_to_contour(filtered_points)

    if filtered_contour is None or len(filtered_points) == 0:
        return filtered_mask, filtered_contour_mask, None, None, []

    cv2.drawContours(filtered_contour_mask, [filtered_contour], -1, 255, 1)

    if len(filtered_contour) >= 3:
        cv2.drawContours(filtered_mask, [filtered_contour], -1, 255, -1)
    else:
        for p in filtered_points:
            row = int(round(p.x))
            col = int(round(p.y))

            if 0 <= row < filtered_mask.shape[0] and 0 <= col < filtered_mask.shape[1]:
                filtered_mask[row, col] = 255

    if len(X_) >= FINAL_POLY_DEGREE + 1 and len(Y_) >= FINAL_POLY_DEGREE + 1:
        final_poly, _ = IdnetifyPoly(X_, Y_, FINAL_POLY_DEGREE)
    else:
        final_poly = None

    return filtered_mask, filtered_contour_mask, filtered_contour, final_poly, filtered_points


def draw_filtered_principal_overlay(crop_gray, contour_mask, final_poly):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    result[contour_mask > 0] = (0, 255, 0)

    if final_poly is not None:
        h, w = result.shape[:2]
        line_points = []

        for col in range(w):
            row = int(round(final_poly(col)))

            if 0 <= row < h:
                line_points.append([col, row])

        if len(line_points) >= 2:
            line_points = np.array(line_points, dtype=np.int32)
            cv2.polylines(result, [line_points], False, (0, 165, 255), 1, cv2.LINE_AA)

    return result


def build_secondary_roi_boxes(image_shape, principal_points, lateral_poly):
    h, w = image_shape[:2]

    invalid = CropBox(0, 0, h, w, False)

    if principal_points is None or len(principal_points) == 0:
        return invalid, invalid

    cols = np.array([p.y for p in principal_points], dtype=np.float32)
    rows = np.array([p.x for p in principal_points], dtype=np.float32)

    if len(cols) == 0 or len(rows) == 0:
        return invalid, invalid

    left_right = int(round(np.min(cols))) + SECONDARY_LEFT_EXTRA_COLS
    right_left = int(round(np.max(cols))) - SECONDARY_RIGHT_EXTRA_COLS

    left_right = max(1, min(left_right, w))
    right_left = max(0, min(right_left, w - 1))

    if lateral_poly is not None and left_right > 1:
        left_cols = np.arange(0, left_right, dtype=np.float32)
        left_pred_rows = lateral_poly(left_cols)
        left_top = int(round(np.min(left_pred_rows))) - SECONDARY_TOP_MARGIN_LEFT
    else:
        left_top = int(round(np.min(rows))) - SECONDARY_TOP_MARGIN_LEFT

    if lateral_poly is not None and right_left < w - 1:
        right_cols = np.arange(right_left, w, dtype=np.float32)
        right_pred_rows = lateral_poly(right_cols)
        right_top = int(round(np.min(right_pred_rows))) - SECONDARY_TOP_MARGIN_RIGHT
    else:
        right_top = int(round(np.min(rows))) - SECONDARY_TOP_MARGIN_RIGHT

    left_box = CropBox(
        top=max(0, left_top),
        left=0,
        bottom=h,
        right=left_right,
        valid=True
    )

    right_box = CropBox(
        top=max(0, right_top),
        left=right_left,
        bottom=h,
        right=w,
        valid=True
    )

    left_box = clamp_box(left_box, image_shape)
    right_box = clamp_box(right_box, image_shape)

    if not is_box_valid(left_box, image_shape):
        left_box.valid = False

    if not is_box_valid(right_box, image_shape):
        right_box.valid = False

    return left_box, right_box


def crop_box_region(image, box):
    if box is None or not box.valid:
        return np.zeros((1, 1), dtype=np.uint8)

    return image[box.top:box.bottom, box.left:box.right].copy()


def draw_secondary_rois_overlay(crop_gray, left_box, right_box, principal_contour_mask):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if principal_contour_mask is not None:
        result[principal_contour_mask > 0] = (0, 255, 0)

    if left_box is not None and left_box.valid:
        cv2.rectangle(
            result,
            (left_box.left, left_box.top),
            (left_box.right - 1, left_box.bottom - 1),
            (255, 0, 0),
            2
        )

    if right_box is not None and right_box.valid:
        cv2.rectangle(
            result,
            (right_box.left, right_box.top),
            (right_box.right - 1, right_box.bottom - 1),
            (0, 0, 255),
            2
        )

    return result


def contour_local_to_global(contour, box):
    if contour is None:
        return None

    result = contour.copy()
    result[:, 0, 0] += box.left
    result[:, 0, 1] += box.top

    return result


def build_secondary_component_from_roi(binary_mask, roi_box, lateral_poly, roi_binary_override=None):
    full_mask = np.zeros(binary_mask.shape[:2], dtype=np.uint8)
    full_contour_mask = np.zeros(binary_mask.shape[:2], dtype=np.uint8)
    local_debug = np.zeros((1, 1, 3), dtype=np.uint8)

    if roi_box is None or not roi_box.valid:
        return full_mask, full_contour_mask, None, local_debug

    if roi_binary_override is not None:
        roi_binary = roi_binary_override.copy()
    else:
        roi_binary = crop_box_region(binary_mask, roi_box)

    if roi_binary is None or roi_binary.size == 0:
        return full_mask, full_contour_mask, None, local_debug

    local_points = []
    ExtractContour(roi_binary, local_points)

    if len(local_points) < SECONDARY_MIN_POINTS:
        local_debug = draw_traveler_points(roi_binary, local_points)
        return full_mask, full_contour_mask, None, local_debug

    distances = removeOutliers(local_points, OUTLIER_DISTANCE)
    selected_points, _, _, scored_components = ExtractBestPleuralTravelerComponent(
        distances,
        local_points,
        roi_binary.shape
    )

    band_points, band_info = filter_travelers_by_main_horizontal_band(
        selected_points,
        roi_binary.shape
    )

    X_ = [p.x for p in band_points]
    Y_ = [p.y for p in band_points]

    if len(band_points) >= POLY_MIN_POINTS:
        local_poly, _ = IdnetifyPoly(X_, Y_, POLY_DEGREE)
        filtered_points = Fit(band_points, local_poly, POLY_DEVIATION)
    else:
        local_poly = None
        filtered_points = band_points

    local_debug = draw_poly_filtered_travelers(
        roi_binary,
        band_points,
        filtered_points,
        local_poly
    )

    if len(filtered_points) < SECONDARY_MIN_POINTS:
        return full_mask, full_contour_mask, None, local_debug

    _, local_roi_box, local_roi = build_principal_roi_from_travelers(
        roi_binary,
        filtered_points
    )

    local_component_mask, local_component_contour_mask, local_contour = (
        extract_largest_component_contour_from_roi(
            local_roi,
            local_roi_box,
            roi_binary.shape
        )
    )

    if local_contour is None:
        return full_mask, full_contour_mask, None, local_debug

    global_contour = contour_local_to_global(local_contour, roi_box)

    if global_contour is None:
        return full_mask, full_contour_mask, None, local_debug

    if lateral_poly is not None:
        global_points = contour_to_points(global_contour)
        filtered_global_points, _, _ = Fit2(
            global_points,
            lateral_poly,
            SECONDARY_CONTOUR_DEVIATION
        )
        filtered_global_contour = points_to_contour(filtered_global_points)
    else:
        filtered_global_contour = global_contour

    if filtered_global_contour is None or len(filtered_global_contour) < 3:
        return full_mask, full_contour_mask, None, local_debug

    cv2.drawContours(full_mask, [filtered_global_contour], -1, 255, -1)
    cv2.drawContours(full_contour_mask, [filtered_global_contour], -1, 255, 1)

    return full_mask, full_contour_mask, filtered_global_contour, local_debug


def draw_secondary_component_overlay(crop_gray, principal_contour_mask, left_contour_mask, right_contour_mask):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if principal_contour_mask is not None:
        result[principal_contour_mask > 0] = (0, 255, 0)

    if left_contour_mask is not None:
        result[left_contour_mask > 0] = (255, 0, 0)

    if right_contour_mask is not None:
        result[right_contour_mask > 0] = (0, 0, 255)

    return result


def contour_is_valid(contour, min_area=MERGE_MIN_COMPONENT_AREA):
    if contour is None:
        return False

    if len(contour) < 3:
        return False

    return cv2.contourArea(contour) >= min_area


def get_endpoint_candidates(contour, side, sample_count=MERGE_ENDPOINT_SAMPLE):
    if contour is None or len(contour) == 0:
        return np.empty((0, 2), dtype=np.int32)

    pts = contour.reshape(-1, 2)

    if side == "left":
        order = np.argsort(pts[:, 0])
    else:
        order = np.argsort(-pts[:, 0])

    sample_count = max(1, min(sample_count, len(order)))
    selected = pts[order[:sample_count]]

    return selected.astype(np.int32)


def find_closest_points(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return None, None

    best_distance = None
    best_a = None
    best_b = None

    for a in points_a:
        diff = points_b.astype(np.float32) - a.astype(np.float32)
        dist = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dist))
        current_distance = float(dist[idx])

        if best_distance is None or current_distance < best_distance:
            best_distance = current_distance
            best_a = a
            best_b = points_b[idx]

    return best_a, best_b


def connect_two_contours(mask, contour_a, side_a, contour_b, side_b):
    points_a = get_endpoint_candidates(contour_a, side_a)
    points_b = get_endpoint_candidates(contour_b, side_b)

    p1, p2 = find_closest_points(points_a, points_b)

    if p1 is None or p2 is None:
        return mask

    cv2.line(
        mask,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        255,
        MERGE_CONNECTION_THICKNESS,
        cv2.LINE_AA
    )

    hull_points = np.vstack([points_a, points_b]).astype(np.int32)

    if len(hull_points) >= 3:
        hull = cv2.convexHull(hull_points.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 255, -1)

    return mask


def merge_pleural_components(shape, principal_contour, left_contour, right_contour):
    merged_mask = np.zeros(shape[:2], dtype=np.uint8)
    merged_contour_mask = np.zeros(shape[:2], dtype=np.uint8)

    if not contour_is_valid(principal_contour, min_area=1):
        return merged_mask, merged_contour_mask, None

    cv2.drawContours(merged_mask, [principal_contour], -1, 255, -1)

    if contour_is_valid(left_contour):
        cv2.drawContours(merged_mask, [left_contour], -1, 255, -1)
        merged_mask = connect_two_contours(
            merged_mask,
            left_contour,
            "right",
            principal_contour,
            "left"
        )

    if contour_is_valid(right_contour):
        cv2.drawContours(merged_mask, [right_contour], -1, 255, -1)
        merged_mask = connect_two_contours(
            merged_mask,
            principal_contour,
            "right",
            right_contour,
            "left"
        )

    kernel = np.ones((3, 3), dtype=np.uint8)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours = get_contours(merged_mask)

    if len(contours) == 0:
        return merged_mask, merged_contour_mask, None

    final_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(merged_contour_mask, [final_contour], -1, 255, 1)

    return merged_mask, merged_contour_mask, final_contour


def draw_final_merged_overlay(crop_gray, merged_contour_mask):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if merged_contour_mask is not None:
        result[merged_contour_mask > 0] = (0, 255, 0)

    return result


def limit_mask_to_poly_band(mask, poly_line, above_px, below_px):
    if mask is None or mask.size == 0:
        return mask

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if poly_line is None:
        return mask.copy()

    h, w = mask.shape[:2]
    limited = np.zeros_like(mask, dtype=np.uint8)

    for col in range(w):
        row_center = int(round(poly_line(col)))

        row_top = max(0, row_center - above_px)
        row_bottom = min(h - 1, row_center + below_px)

        if row_top <= row_bottom:
            limited[row_top:row_bottom + 1, col] = mask[row_top:row_bottom + 1, col]

    return limited


def keep_largest_component(mask):
    if mask is None or mask.size == 0:
        return mask

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)

    if np.count_nonzero(binary) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest_label] = 255

    return out


def limit_secondary_mask_by_global_poly(binary_mask, roi_box, lateral_poly):
    if roi_box is None or not roi_box.valid:
        return np.zeros((1, 1), dtype=np.uint8)

    roi = crop_box_region(binary_mask, roi_box)

    if roi is None or roi.size == 0:
        return roi

    if lateral_poly is None:
        return roi

    h, w = roi.shape[:2]
    limited = np.zeros_like(roi, dtype=np.uint8)

    for local_col in range(w):
        global_col = roi_box.left + local_col
        global_row_center = int(round(lateral_poly(global_col)))
        local_row_center = global_row_center - roi_box.top

        row_top = max(0, local_row_center - SECONDARY_POLY_BAND_ABOVE_PX)
        row_bottom = min(h - 1, local_row_center + SECONDARY_POLY_BAND_BELOW_PX)

        if row_top <= row_bottom:
            limited[row_top:row_bottom + 1, local_col] = roi[row_top:row_bottom + 1, local_col]

    if SECONDARY_KEEP_LARGEST_IN_BAND:
        limited = keep_largest_component(limited)

    return limited


def draw_secondary_band_debug(crop_gray, left_box, right_box, final_poly):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if final_poly is None:
        return result

    h, w = result.shape[:2]

    for box, color in [(left_box, (255, 0, 0)), (right_box, (0, 0, 255))]:
        if box is None or not box.valid:
            continue

        center_points = []
        top_points = []
        bottom_points = []

        for col in range(box.left, box.right):
            row_center = int(round(final_poly(col)))
            row_top = row_center - SECONDARY_POLY_BAND_ABOVE_PX
            row_bottom = row_center + SECONDARY_POLY_BAND_BELOW_PX

            if 0 <= row_center < h:
                center_points.append([col, row_center])
            if 0 <= row_top < h:
                top_points.append([col, row_top])
            if 0 <= row_bottom < h:
                bottom_points.append([col, row_bottom])

        if len(center_points) >= 2:
            cv2.polylines(result, [np.array(center_points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        if len(top_points) >= 2:
            cv2.polylines(result, [np.array(top_points, dtype=np.int32)], False, (0, 165, 255), 1, cv2.LINE_AA)
        if len(bottom_points) >= 2:
            cv2.polylines(result, [np.array(bottom_points, dtype=np.int32)], False, (0, 165, 255), 1, cv2.LINE_AA)

    return result


def limit_pleura_thickness_by_poly(mask, poly_line):
    limited_mask = np.zeros_like(mask, dtype=np.uint8)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)

    if mask is None or mask.size == 0:
        return limited_mask, contour_mask, None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if poly_line is None:
        contours = get_contours(mask)
        if len(contours) == 0:
            return mask.copy(), contour_mask, None

        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_mask, [contour], -1, 255, 1)
        return mask.copy(), contour_mask, contour

    h, w = mask.shape[:2]

    for col in range(w):
        row_center = int(round(poly_line(col)))

        row_top = max(0, row_center - PLEURA_THICKNESS_ABOVE_PX)
        row_bottom = min(h - 1, row_center + PLEURA_THICKNESS_BELOW_PX)

        if row_top <= row_bottom:
            limited_mask[row_top:row_bottom + 1, col] = mask[row_top:row_bottom + 1, col]

    if PLEURA_LIMIT_KEEP_LARGEST:
        contours = get_contours(limited_mask)

        if len(contours) == 0:
            return limited_mask, contour_mask, None

        contour = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(limited_mask, dtype=np.uint8)
        cv2.drawContours(clean, [contour], -1, 255, -1)
        limited_mask = clean
    else:
        contours = get_contours(limited_mask)

        if len(contours) == 0:
            return limited_mask, contour_mask, None

        contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(contour_mask, [contour], -1, 255, 1)

    return limited_mask, contour_mask, contour


def draw_limited_pleura_overlay(crop_gray, limited_contour_mask, poly_line):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()

    if limited_contour_mask is not None:
        result[limited_contour_mask > 0] = (0, 255, 0)

    if poly_line is not None:
        h, w = result.shape[:2]
        center_points = []
        top_points = []
        bottom_points = []

        for col in range(w):
            row_center = int(round(poly_line(col)))
            row_top = row_center - PLEURA_THICKNESS_ABOVE_PX
            row_bottom = row_center + PLEURA_THICKNESS_BELOW_PX

            if 0 <= row_center < h:
                center_points.append([col, row_center])

            if 0 <= row_top < h:
                top_points.append([col, row_top])

            if 0 <= row_bottom < h:
                bottom_points.append([col, row_bottom])

        if len(center_points) >= 2:
            cv2.polylines(result, [np.array(center_points, dtype=np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)

        if len(top_points) >= 2:
            cv2.polylines(result, [np.array(top_points, dtype=np.int32)], False, (0, 165, 255), 1, cv2.LINE_AA)

        if len(bottom_points) >= 2:
            cv2.polylines(result, [np.array(bottom_points, dtype=np.int32)], False, (0, 165, 255), 1, cv2.LINE_AA)

    return result


def debug_print(*args):
    if DEBUG_CONSOLE:
        print(*args)


def summarize_points(points):
    if points is None or len(points) == 0:
        return "count=0"

    rows = np.array([p.x for p in points], dtype=np.float32)
    cols = np.array([p.y for p in points], dtype=np.float32)

    return (
        f"count={len(points)} "
        f"x=[{int(np.min(rows))},{int(np.max(rows))}] "
        f"y=[{int(np.min(cols))},{int(np.max(cols))}]"
    )


def summarize_box(name, box):
    if box is None:
        return f"{name}: None"

    return (
        f"{name}: valid={box.valid} "
        f"top={box.top} left={box.left} bottom={box.bottom} right={box.right} "
        f"size=({max(0, box.bottom - box.top)}, {max(0, box.right - box.left)})"
    )


def summarize_contour(name, contour):
    if contour is None or len(contour) == 0:
        return f"{name}: absent"

    area = float(cv2.contourArea(contour))
    pts = contour.reshape(-1, 2)
    xmin = int(np.min(pts[:, 0]))
    xmax = int(np.max(pts[:, 0]))
    ymin = int(np.min(pts[:, 1]))
    ymax = int(np.max(pts[:, 1]))

    return (
        f"{name}: points={len(pts)} area={area:.1f} "
        f"x=[{xmin},{xmax}] y=[{ymin},{ymax}] "
        f"w={xmax - xmin + 1} h={ymax - ymin + 1}"
    )


def summarize_scored_components(scored_components, top_k=5):
    if scored_components is None or len(scored_components) == 0:
        return ["fara componente"]

    ordered = sorted(scored_components, key=lambda item: item["score"], reverse=True)
    lines = []

    for idx, item in enumerate(ordered[:top_k], start=1):
        info = item["info"]
        lines.append(
            f"#{idx} score={item['score']:.3f} "
            f"count={info['count']} "
            f"width_frac={info['width_frac']:.3f} "
            f"height_frac={info['height_frac']:.3f} "
            f"median_row_frac={info['median_row_frac']:.3f} "
            f"aspect={info['aspect']:.2f} "
            f"slope={info['slope']:.3f}"
        )

    return lines



def save_debug_image(index, step_number, name, image):
    if not DEBUG_SAVE_IMPORTANT_STEPS:
        return

    ensure_dir(DEBUG_IMPORTANT_STEPS_DIR)

    if image is None:
        return

    base = f"{index:02d}_{step_number:02d}_{name}.png"
    path = os.path.join(DEBUG_IMPORTANT_STEPS_DIR, base)
    cv2.imwrite(path, image)


def append_identification_report(row):
    if not WRITE_IDENTIFICATION_REPORT:
        return

    ensure_dir(OUTPUT_DIR)

    file_exists = os.path.exists(IDENTIFICATION_REPORT_PATH)

    fieldnames = [
        "index",
        "threshold",
        "white_pixels_binary",
        "traveler_points",
        "selected_points",
        "band_points",
        "filtered_points",
        "principal_roi_valid",
        "principal_roi_white_pixels",
        "principal_contour_area",
        "filtered_principal_area",
        "left_secondary_area",
        "right_secondary_area",
        "merged_area",
        "limited_area",
        "has_final_contour"
    ]

    with open(IDENTIFICATION_REPORT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def contour_area_safe(contour):
    if contour is None or len(contour) == 0:
        return 0.0

    return float(cv2.contourArea(contour))


def save_important_identification_steps(
    index,
    crop,
    palette,
    binary,
    traveler_points,
    selected_points,
    band_points,
    band_info,
    filtered_points,
    poly_line,
    roi_box,
    principal_component_contour,
    filtered_principal_contour,
    final_poly,
    left_secondary_contour,
    right_secondary_contour,
    merged_pleura_contour,
    limited_pleura_contour
):
    if not DEBUG_SAVE_IMPORTANT_STEPS:
        return

    save_debug_image(index, 1, "crop", crop)
    save_debug_image(index, 2, "palette_7", palette)
    save_debug_image(index, 3, "binary_mask", binary)

    save_debug_image(
        index,
        4,
        "traveler_points",
        draw_traveler_points(crop, traveler_points)
    )

    save_debug_image(
        index,
        5,
        "selected_component",
        draw_selected_pleural_travelers(crop, traveler_points, selected_points)
    )

    save_debug_image(
        index,
        6,
        "band_filtered",
        draw_band_filtered_travelers(crop, selected_points, band_points, band_info)
    )

    save_debug_image(
        index,
        7,
        "poly_filtered",
        draw_poly_filtered_travelers(crop, band_points, filtered_points, poly_line)
    )

    save_debug_image(
        index,
        8,
        "principal_roi",
        draw_principal_roi_overlay(crop, roi_box, filtered_points)
    )

    save_debug_image(
        index,
        9,
        "principal_component",
        draw_principal_component_overlay(crop, principal_component_contour)
    )

    save_debug_image(
        index,
        10,
        "filtered_principal",
        draw_filtered_principal_overlay(crop, filtered_principal_contour, final_poly)
    )

    save_debug_image(
        index,
        11,
        "secondary_components",
        draw_secondary_component_overlay(
            crop,
            filtered_principal_contour,
            left_secondary_contour,
            right_secondary_contour
        )
    )

    save_debug_image(
        index,
        12,
        "merged_pleura",
        draw_final_merged_overlay(crop, merged_pleura_contour)
    )

    save_debug_image(
        index,
        13,
        "limited_final",
        draw_limited_pleura_overlay(crop, limited_pleura_contour, final_poly)
    )

def save_final_contour_on_original(index, image_bgr, crop_box, final_contour_mask):
    ensure_dir(FINAL_CONTOUR_ON_ORIGINAL_DIR)

    if image_bgr is None:
        return

    result = image_bgr.copy()

    if final_contour_mask is not None and final_contour_mask.size > 0:
        contours = get_contours(final_contour_mask)

        for contour in contours:
            contour_on_original = contour.copy()
            contour_on_original[:, 0, 0] += crop_box.left
            contour_on_original[:, 0, 1] += crop_box.top

            cv2.drawContours(
                result,
                [contour_on_original],
                -1,
                (0, 255, 0),
                FINAL_CONTOUR_THICKNESS
            )

    base = f"{index:02d}"
    output_path = os.path.join(
        FINAL_CONTOUR_ON_ORIGINAL_DIR,
        base + "_final_contour_on_original.png"
    )

    cv2.imwrite(output_path, result)

def save_crop(index):
    path = find_image_path(index)

    if path is None:
        debug_print(f"[{index}] Imagine inexistenta.")
        return False

    debug_print("")
    debug_print("=" * 70)
    debug_print(f"Procesez imaginea: {index}")

    image_bgr = read_image_bgr(path)
    crop, crop_box = crop_border(image_bgr)
    palette = reduce_palette_7(crop)
    binary, threshold = binarize_palette_7(palette)

    debug_print(summarize_box("Crop box", crop_box))
    debug_print(f"Binary threshold(top1) = {threshold}")
    debug_print(f"Pixeli albi in masca binara: {int(np.count_nonzero(binary))}")

    traveler_points = []
    ExtractContour(binary, traveler_points)
    debug_print("Traveler points:", summarize_points(traveler_points))

    distances = removeOutliers(traveler_points, OUTLIER_DISTANCE)
    selected_points, _, _, scored_components = ExtractBestPleuralTravelerComponent(
        distances,
        traveler_points,
        binary.shape
    )

    debug_print(f"Componente travelers gasite: {len(scored_components)}")
    for line in summarize_scored_components(scored_components, top_k=5):
        debug_print(" ", line)
    debug_print("Componenta selectata:", summarize_points(selected_points))

    band_points, band_info = filter_travelers_by_main_horizontal_band(
        selected_points,
        binary.shape
    )

    debug_print("Dupa filtrare pe banda:", summarize_points(band_points))
    if band_info is not None:
        debug_print(
            f"Banda: center={band_info['band_center']} "
            f"half_height={band_info['half_height']} "
            f"fallback={band_info['used_fallback']}"
        )

    X_ = [p.x for p in band_points]
    Y_ = [p.y for p in band_points]

    if len(band_points) >= POLY_MIN_POINTS:
        poly_line, _ = IdnetifyPoly(X_, Y_, POLY_DEGREE)
        filtered_points = Fit(band_points, poly_line, POLY_DEVIATION)
    else:
        poly_line = None
        filtered_points = band_points

    debug_print("Dupa fit polinomial:", summarize_points(filtered_points))
    debug_print(f"Polinom initial valid: {poly_line is not None}")

    binary_cut, roi_box, principal_roi = build_principal_roi_from_travelers(
        binary,
        filtered_points
    )

    debug_print(summarize_box("Principal ROI", roi_box))
    debug_print(f"Principal ROI pixeli albi: {int(np.count_nonzero(principal_roi))}")

    principal_component_mask, principal_component_contour, principal_contour = (
        extract_largest_component_contour_from_roi(
            principal_roi,
            roi_box,
            binary.shape
        )
    )

    debug_print(summarize_contour("Principal contour brut", principal_contour))

    filtered_principal_mask, filtered_principal_contour, filtered_principal_ctr, final_poly, final_points = (
        build_filtered_principal_component(
            principal_contour,
            poly_line,
            binary.shape
        )
    )

    debug_print(summarize_contour("Principal contour filtrat", filtered_principal_ctr))
    debug_print("Puncte finale principal:", summarize_points(final_points))
    debug_print(f"Polinom final valid: {final_poly is not None}")

    left_secondary_box, right_secondary_box = build_secondary_roi_boxes(
        binary.shape,
        final_points,
        final_poly
    )

    debug_print(summarize_box("Secondary ROI LEFT", left_secondary_box))
    debug_print(summarize_box("Secondary ROI RIGHT", right_secondary_box))

    left_secondary_roi_raw = crop_box_region(binary, left_secondary_box)
    right_secondary_roi_raw = crop_box_region(binary, right_secondary_box)

    left_secondary_roi = limit_secondary_mask_by_global_poly(
        binary,
        left_secondary_box,
        final_poly
    )
    right_secondary_roi = limit_secondary_mask_by_global_poly(
        binary,
        right_secondary_box,
        final_poly
    )

    debug_print(f"Pixeli albi left ROI raw: {int(np.count_nonzero(left_secondary_roi_raw))}")
    debug_print(f"Pixeli albi right ROI raw: {int(np.count_nonzero(right_secondary_roi_raw))}")
    debug_print(f"Pixeli albi left ROI limitat: {int(np.count_nonzero(left_secondary_roi))}")
    debug_print(f"Pixeli albi right ROI limitat: {int(np.count_nonzero(right_secondary_roi))}")

    left_secondary_mask, left_secondary_contour, left_secondary_ctr, left_secondary_debug = (
        build_secondary_component_from_roi(
            binary,
            left_secondary_box,
            final_poly,
            roi_binary_override=left_secondary_roi
        )
    )

    right_secondary_mask, right_secondary_contour, right_secondary_ctr, right_secondary_debug = (
        build_secondary_component_from_roi(
            binary,
            right_secondary_box,
            final_poly,
            roi_binary_override=right_secondary_roi
        )
    )

    debug_print(summarize_contour("Left secondary contour", left_secondary_ctr))
    debug_print(summarize_contour("Right secondary contour", right_secondary_ctr))

    merged_pleura_mask, merged_pleura_contour, merged_pleura_ctr = merge_pleural_components(
        binary.shape,
        filtered_principal_ctr,
        left_secondary_ctr,
        right_secondary_ctr
    )

    limited_pleura_mask, limited_pleura_contour, limited_pleura_ctr = limit_pleura_thickness_by_poly(
        merged_pleura_mask,
        final_poly
    )

    debug_print(summarize_contour("Merged pleura contour", merged_pleura_ctr))
    debug_print(f"Pixeli albi merged mask: {int(np.count_nonzero(merged_pleura_mask))}")
    debug_print(summarize_contour("Limited pleura contour", limited_pleura_ctr))
    debug_print(f"Pixeli albi limited mask: {int(np.count_nonzero(limited_pleura_mask))}")

    if limited_pleura_ctr is None:
        debug_print("ATENTIE: Nu s-a obtinut contur final dupa limitarea grosimii.")
    elif filtered_principal_ctr is None:
        debug_print("ATENTIE: Lipseste componenta principala, deci unirea nu are baza.")
    else:
        debug_print("Rezultat final: contur final limitat obtinut.")

    ensure_dir(OUTPUT_DIR)

    save_important_identification_steps(
        index=index,
        crop=crop,
        palette=palette,
        binary=binary,
        traveler_points=traveler_points,
        selected_points=selected_points,
        band_points=band_points,
        band_info=band_info,
        filtered_points=filtered_points,
        poly_line=poly_line,
        roi_box=roi_box,
        principal_component_contour=principal_component_contour,
        filtered_principal_contour=filtered_principal_contour,
        final_poly=final_poly,
        left_secondary_contour=left_secondary_contour,
        right_secondary_contour=right_secondary_contour,
        merged_pleura_contour=merged_pleura_contour,
        limited_pleura_contour=limited_pleura_contour
    )

    append_identification_report({
        "index": index,
        "threshold": threshold,
        "white_pixels_binary": int(np.count_nonzero(binary)),
        "traveler_points": len(traveler_points),
        "selected_points": len(selected_points),
        "band_points": len(band_points),
        "filtered_points": len(filtered_points),
        "principal_roi_valid": bool(roi_box.valid),
        "principal_roi_white_pixels": int(np.count_nonzero(principal_roi)),
        "principal_contour_area": contour_area_safe(principal_contour),
        "filtered_principal_area": contour_area_safe(filtered_principal_ctr),
        "left_secondary_area": contour_area_safe(left_secondary_ctr),
        "right_secondary_area": contour_area_safe(right_secondary_ctr),
        "merged_area": contour_area_safe(merged_pleura_ctr),
        "limited_area": contour_area_safe(limited_pleura_ctr),
        "has_final_contour": limited_pleura_ctr is not None
    })

    save_final_contour_on_original(
        index,
        image_bgr,
        crop_box,
        limited_pleura_contour
    )

    return True


def main_batch():
    if RESET_OUTPUT_DIR_ON_RUN:
        reset_output_dir(OUTPUT_DIR)
    else:
        ensure_dir(OUTPUT_DIR)

    for index in range(START_IDX, END_IDX):
        save_crop(index)


def main_single():
    if RESET_OUTPUT_DIR_ON_RUN:
        reset_output_dir(OUTPUT_DIR)
    else:
        ensure_dir(OUTPUT_DIR)

    save_crop(SINGLE_IMAGE_IDX)


if __name__ == "__main__":
    if RUN_SINGLE_IMAGE:
        main_single()
    else:
        main_batch()
