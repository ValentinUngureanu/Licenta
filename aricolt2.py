import os
import shutil
from dataclasses import dataclass
import cv2
import numpy as np
INPUT_DIR = 'C:\\Facultate\\AN4\\Licenta\\Licenta-Cod\\ORIGINAL_IMAGES'
OUTPUT_DIR = 'C:\\Facultate\\AN4\\Licenta\\Licenta-Cod\\CROP_RESULTS'
FINAL_CONTOUR_ON_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, 'FINAL_CONTOUR_ON_ORIGINAL')
DEBUG_IMPORTANT_STEPS_DIR = os.path.join(OUTPUT_DIR, 'DEBUG_IMPORTANT_STEPS')
RESET_OUTPUT_DIR_ON_RUN = True
START_IDX = 0
END_IDX = 61
SINGLE_IMAGE_IDX = 1
RUN_SINGLE_IMAGE = False
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
MAX_PIXEL_VALUE = 255
DEFAULT_BINARY_THRESHOLD = 128
BINARY_THRESHOLD = 110
MIN_BORDER_AREA_FRAC = 0.0007
SMALL_CONTOUR_AREA_FRAC = 3e-05
LEFT_MARGIN_SEARCH_FRAC = 0.07
DEFAULT_LEFT_BOUND_FRAC = 0.018
CROP_RIGHT_MARGIN_FRAC = 0.014
MIN_CROP_SIZE_FRAC = 0.1
HORIZONTAL_KERNEL_WIDTH_FRAC = 0.0015
MORPH_ITERATIONS = 3
SUSPECT_FULL_CROP_FRAC = 0.9
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
HORIZONTAL_SMOOTH_FRAC = 0.03
HORIZONTAL_THRESHOLD_PERCENTILES = [45, 50, 55, 60, 65, 70, 75]
HORIZONTAL_THRESHOLD_SCALE = 0.65
PALETTE_METHOD = 'clahe_kmeans'
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
POLY_DEVIATION = 35
POLY_MIN_POINTS = 10
COMPONENT_MIN_POINTS = 8
COMPONENT_MIN_WIDTH_FRAC = 0.035
COMPONENT_TOO_LOW_FRAC = 0.7
COMPONENT_TOO_HIGH_FRAC = 0.06
BAND_BIN_HEIGHT_FRAC = 0.018
BAND_KEEP_HALF_HEIGHT_FRAC = 0.045
BAND_MIN_POINTS = 8
ROI_PAD_X = 10
ROI_PAD_Y_TOP = 10
ROI_PAD_Y_BOTTOM = 10
PRINCIPAL_ROI_MIN_HEIGHT_FRAC = 0.025
PRINCIPAL_ROI_MIN_WIDTH_FRAC = 0.05
REMOVE_BELOW_TRAVELERS_OFFSET = 10
PRINCIPAL_COMPONENT_MIN_AREA_FRAC = 1e-05
PRINCIPAL_COMPONENT_DILATE_KERNEL = 3
PRINCIPAL_CONTOUR_DEVIATION = 50
FINAL_POLY_DEGREE = 2
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
PLEURA_THICKNESS_ABOVE_PX = 18
PLEURA_THICKNESS_BELOW_PX = 22
PLEURA_LIMIT_KEEP_LARGEST = True
FINAL_CONTOUR_THICKNESS = 1
CURVED_BAND_POLY_DEGREE = 2
CURVED_BAND_MIN_POINTS = 20
CURVED_BAND_ABOVE_PX = 25
CURVED_BAND_BELOW_PX = 35
CURVED_MIN_WIDTH_FRAC = 0.1
CURVED_MIN_SCORE_GAIN = 1.12
HORIZONTAL_GOOD_KEEP_RATIO = 0.6
HORIZONTAL_GOOD_WIDTH_FRAC = 0.35
FORCE_CURVED_ROW_SPAN_FRAC = 0.075
FORCE_CURVED_SLOPE_ABS = 0.18
CURVED_BAND_MIN_KEEP_POINTS = 10
CURVED_BAND_MIN_MIDPOINT_COLUMNS = 12
CURVED_BAND_MIDPOINT_SMOOTH_FRAC = 0.035
CURVED_BAND_MAX_POLY_AMPLITUDE_FRAC = 0.18
CURVED_BAND_MAX_RESIDUAL_FRAC = 0.075
CURVED_BAND_REJECT_IF_EDGE_EXPLODES_FRAC = 0.08
CURVED_LOCAL_ENABLE = True
CURVED_LOCAL_MIN_DENSE_WIDTH_FRAC = 0.12
CURVED_LOCAL_MAX_COLUMN_GAP_FRAC = 0.035
CURVED_LOCAL_MIN_DENSE_COLUMNS = 14
CURVED_LOCAL_SMOOTH_FRAC = 0.045
CURVED_LOCAL_EXTENSION_FRAC = 0.05
CURVED_LOCAL_EXTENSION_MAX_SLOPE = 0.05
CURVED_LOCAL_MAX_AMPLITUDE_FRAC = 0.16
CURVED_LOCAL_MAX_RESIDUAL_FRAC = 0.065
CURVED_LOCAL_EDGE_JUMP_FRAC = 0.075
CURVED_ACCEPT_MIN_KEEP_RATIO = 0.38
CURVED_ACCEPT_MIN_CONTINUITY = 0.18
CURVED_ACCEPT_MAX_HEIGHT_FRAC = 0.22
CURVED_ACCEPT_IF_HORIZONTAL_WIDTH_LOSS_FRAC = 0.16
CURVED_ACCEPT_IF_HORIZONTAL_KEEP_LOSS = 0.18
CURVED_ALLOW_POLYNOMIAL_BACKUP = False
INITIAL_POLY_HORIZONTAL_USE_BAND_CENTER = True
INITIAL_POLY_MAX_AMPLITUDE_FRAC = 0.16
INITIAL_POLY_MAX_RESIDUAL_FRAC = 0.08
INITIAL_POLY_MAX_AMPLITUDE_FRAC_LINEAR = 0.50
FINAL_POLY_MAX_AMPLITUDE_FRAC_LINEAR = 0.50

# Banda liniara robusta este fallback-ul sigur dintre orizontal si curbat.
# Ajuta in imaginile unde pleura este oblica, iar banda orizontala alege doar
# un fragment local mai jos/sus. Nu inlocuieste conturul cu linie artificiala,
# ci doar alege ROI-ul si banda in care se pastreaza masca reala.
LINEAR_BAND_ENABLE = True
LINEAR_BAND_MIN_POINTS = 10
LINEAR_BAND_MIN_WIDTH_FRAC = 0.12
LINEAR_BAND_ABOVE_PX = 25
LINEAR_BAND_BELOW_PX = 35
LINEAR_BAND_MAX_ITER = 4
LINEAR_BAND_MAX_SLOPE_ABS = 0.75
LINEAR_BAND_MAX_HEIGHT_FRAC = 0.50
LINEAR_ACCEPT_MIN_KEEP_RATIO = 0.24
LINEAR_ACCEPT_MIN_CONTINUITY = 0.10
LINEAR_ACCEPT_SCORE_GAIN = 0.92
LINEAR_ACCEPT_IF_HORIZONTAL_WIDTH_LOSS_FRAC = 0.08
LINEAR_ACCEPT_IF_HORIZONTAL_KEEP_LOSS = 0.06

FINAL_POLY_DEGREE_HORIZONTAL = 1
FINAL_POLY_DEGREE_CURVED = 2
FINAL_POLY_MAX_AMPLITUDE_FRAC = 0.2

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

class LocalCurveModel:
    """
    Model de curba locala stabila pentru banda pleurei.

    Foloseste doar intervalul unde exista puncte reale suficient de dense.
    In afara intervalului dens permite doar o extensie mica, cu panta limitata.
    Punctele aflate mai departe de aceasta extensie sunt ignorate de filtrarea pe banda.
    """

    def __init__(self, cols, rows, image_shape, extension_frac, max_slope):
        self.cols = np.asarray(cols, dtype=np.float32)
        self.rows = np.asarray(rows, dtype=np.float32)
        self.image_shape = image_shape
        h, w = image_shape[:2]

        self.col_min = float(np.min(self.cols))
        self.col_max = float(np.max(self.cols))
        self.extension = float(max(0.0, extension_frac) * max(w, 1))
        self.active_col_min = max(0.0, self.col_min - self.extension)
        self.active_col_max = min(float(w - 1), self.col_max + self.extension)

        if len(self.cols) >= 2:
            left_dx = max(float(self.cols[1] - self.cols[0]), 1.0)
            right_dx = max(float(self.cols[-1] - self.cols[-2]), 1.0)
            left_slope = float((self.rows[1] - self.rows[0]) / left_dx)
            right_slope = float((self.rows[-1] - self.rows[-2]) / right_dx)
        else:
            left_slope = 0.0
            right_slope = 0.0

        self.left_slope = float(np.clip(left_slope, -max_slope, max_slope))
        self.right_slope = float(np.clip(right_slope, -max_slope, max_slope))
        self.h = h

    def is_col_allowed(self, col):
        return self.active_col_min <= float(col) <= self.active_col_max

    def __call__(self, col):
        scalar_input = np.isscalar(col)
        arr = np.asarray(col, dtype=np.float32)
        arr_flat = arr.reshape(-1)

        clipped = np.clip(arr_flat, self.col_min, self.col_max)
        values = np.interp(clipped, self.cols, self.rows).astype(np.float32)

        left_mask = arr_flat < self.col_min
        right_mask = arr_flat > self.col_max

        if np.any(left_mask):
            dist = np.minimum(self.col_min - arr_flat[left_mask], self.extension)
            values[left_mask] = self.rows[0] - dist * self.left_slope

        if np.any(right_mask):
            dist = np.minimum(arr_flat[right_mask] - self.col_max, self.extension)
            values[right_mask] = self.rows[-1] + dist * self.right_slope

        values = np.clip(values, 0, max(self.h - 1, 0))

        if scalar_input:
            return float(values[0])

        return values.reshape(arr.shape)

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
        raise ValueError('Nu pot citi imaginea: ' + path)
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
    return (best_start, best_end)

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
        approx = cv2.approxPolyDP(cnt, 1e-05 * peri, True)
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
    box = CropBox(top=int(bar_pixels[0]), left=int(left_bound), bottom=int(bar_pixels[-1]), right=int(bar_pos - crop_right_margin), valid=True)
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
    col_thr = max(0.02, float(np.percentile(col_smooth, 65)) * 0.45)
    row_segment = largest_true_segment(row_smooth > row_thr, int(0.25 * h))
    col_segment = largest_true_segment(col_smooth > col_thr, int(0.35 * w))
    if row_segment is None or col_segment is None:
        return CropBox(0, 0, h, w, False)
    top, bottom = row_segment
    left, right = col_segment
    pad_y = int(0.015 * h)
    pad_x = int(0.015 * w)
    box = CropBox(top=max(0, top - pad_y), left=max(0, left - pad_x), bottom=min(h, bottom + pad_y), right=min(w, right + pad_x), valid=True)
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
        area_frac = box.width * box.height / max(h * w, 1)
        center_y = (box.top + box.bottom) / 2.0 / max(h, 1)
        center_x = (box.left + box.right) / 2.0 / max(w, 1)
        score = 0.0
        score += 2.0 * min(1.0, box.width / max(0.55 * w, 1))
        score += 2.0 * min(1.0, box.height / max(0.45 * h, 1))
        score -= 1.2 * abs(area_frac - 0.55)
        score -= 0.6 * abs(center_x - 0.5)
        if 0.25 <= center_y <= 0.7:
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
    signal = 0.8 * non_black_density + 0.2 * edge_density
    smooth_w = max(9, int(round(HORIZONTAL_SMOOTH_FRAC * w)))
    if smooth_w % 2 == 0:
        smooth_w += 1
    signal_smooth = cv2.blur(signal.astype(np.float32).reshape(1, -1), (smooth_w, 1)).reshape(-1)
    candidates = []
    min_len = max(20, int(round(HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC * w)))
    for perc in HORIZONTAL_THRESHOLD_PERCENTILES:
        base = float(np.percentile(signal_smooth, perc))
        thr = max(0.01, base * HORIZONTAL_THRESHOLD_SCALE)
        segment = largest_true_segment(signal_smooth > thr, min_len)
        if segment is None:
            continue
        left, right = segment
        pad = max(4, int(round(HORIZONTAL_PAD_FRAC * w)))
        left = max(0, left - pad)
        right = min(w, right + pad)
        width = max(1, right - left)
        width_frac = width / max(w, 1)
        center_frac = (left + right) / 2.0 / max(w, 1)
        if width_frac < HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC:
            continue
        score = 0.0
        score += 2.5 * min(1.0, width_frac / 0.75)
        score -= 1.2 * abs(center_frac - 0.5)
        score -= 2.0 * max(0.0, width_frac - HORIZONTAL_MAX_FULL_WIDTH_FRAC)
        score += 0.01 * perc
        candidates.append((score, left, right))
    if len(candidates) == 0:
        thr = max(0.01, float(np.median(signal_smooth)))
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
        final_box = CropBox(top=vertical_box.top, left=horizontal_box.left, bottom=vertical_box.bottom, right=horizontal_box.right, valid=True)
    else:
        final_box = CropBox(top=vertical_box.top, left=0, bottom=vertical_box.bottom, right=w, valid=vertical_box.valid)
    final_box = clamp_box(final_box, gray.shape)
    add_top = max(EXTRA_CROP_TOP_PX, int(round(EXTRA_CROP_TOP_FRAC * final_box.height)))
    add_bottom = max(EXTRA_CROP_BOTTOM_PX, int(round(EXTRA_CROP_BOTTOM_FRAC * final_box.height)))
    add_left = max(EXTRA_CROP_LEFT_PX, int(round(EXTRA_CROP_LEFT_FRAC * final_box.width)))
    add_right = max(EXTRA_CROP_RIGHT_PX, int(round(EXTRA_CROP_RIGHT_FRAC * final_box.width)))
    final_box = CropBox(top=final_box.top + add_top, left=final_box.left + add_left, bottom=final_box.bottom - add_bottom, right=final_box.right - add_right, valid=final_box.valid)
    final_box = clamp_box(final_box, gray.shape)
    crop = gray[final_box.top:final_box.bottom, final_box.left:final_box.right].copy()
    return (crop, final_box)

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
    clahe = cv2.createCLAHE(clipLimit=float(PALETTE_CLAHE_CLIP_LIMIT), tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    valid_low = np.percentile(enhanced, PALETTE_VALID_LOW_PERCENTILE)
    valid_high = np.percentile(enhanced, PALETTE_VALID_HIGH_PERCENTILE)
    valid_mask = (enhanced >= valid_low) & (enhanced <= valid_high)
    samples = enhanced[valid_mask].reshape(-1, 1).astype(np.float32)
    if samples.shape[0] < colors * 20:
        return reduce_palette_7_percentile_linear(gray, colors=colors)
    cv2.setRNGSeed(12345)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(PALETTE_KMEANS_MAX_ITER), 0.4)
    _, labels, centers = cv2.kmeans(samples, int(colors), None, criteria, int(PALETTE_KMEANS_ATTEMPTS), cv2.KMEANS_PP_CENTERS)
    centers = centers.reshape(-1).astype(np.float32)
    order = np.argsort(centers)
    centers_sorted = centers[order]
    output_levels = np.linspace(0, 255, int(colors)).astype(np.uint8)
    flat = enhanced.reshape(-1).astype(np.float32)
    distances = np.abs(flat[:, None] - centers_sorted[None, :])
    nearest = np.argmin(distances, axis=1)
    result = output_levels[nearest].reshape(enhanced.shape).astype(np.uint8)
    dark_cut = max(3, np.percentile(gray, 0.5))
    result[gray <= dark_cut] = 0
    return result

def reduce_palette_7(gray, colors=7):
    if PALETTE_METHOD == 'clahe_kmeans':
        return reduce_palette_7_clahe_kmeans(gray, colors=colors)
    return reduce_palette_7_percentile_linear(gray, colors=colors)

def binarize_palette_7(palette_gray, keep_top_levels=BINARY_KEEP_TOP_LEVELS):
    if palette_gray.ndim == 3:
        palette_gray = cv2.cvtColor(palette_gray, cv2.COLOR_BGR2GRAY)
    palette_gray = palette_gray.astype(np.uint8)
    values = np.sort(np.unique(palette_gray))
    if len(values) < 2:
        return (np.zeros_like(palette_gray, dtype=np.uint8), 0)
    keep_top_levels = max(1, min(keep_top_levels, len(values)))
    threshold = int(values[-keep_top_levels])
    binary = (palette_gray >= threshold).astype(np.uint8) * 255
    return (binary, threshold)

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
    return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))

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
        return (-1000000000.0, {})
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
    info = {'score': float(score), 'count': int(len(component)), 'row_min': row_min, 'row_max': row_max, 'col_min': col_min, 'col_max': col_max, 'width_frac': float(width_frac), 'height_frac': float(height_frac), 'median_row_frac': float(median_row_frac), 'aspect': float(aspect), 'slope': float(slope)}
    return (score, info)

def ExtractBestPleuralTravelerComponent(distances, points, image_shape):
    components = extract_all_traveler_components(distances, points)
    if len(components) == 0:
        return ([], [], [], [])
    scored_components = []
    for idx, component in enumerate(components):
        score, info = score_traveler_component(component, image_shape)
        info['component_index'] = int(idx)
        scored_components.append({'points': component, 'score': float(score), 'info': info})
    selected = max(scored_components, key=lambda item: item['score'])
    selected_points = selected['points']
    X_ = [p.x for p in selected_points]
    Y_ = [p.y for p in selected_points]
    return (selected_points, X_, Y_, scored_components)

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
        return ([], None)
    h, w = image_shape[:2]
    rows = np.array([p.x for p in points], dtype=np.float32)
    bin_height = max(3, int(round(BAND_BIN_HEIGHT_FRAC * h)))
    half_height = max(5, int(round(BAND_KEEP_HALF_HEIGHT_FRAC * h)))
    bins = np.arange(0, h + bin_height, bin_height)
    if len(bins) < 2:
        return (points, None)
    hist, edges = np.histogram(rows, bins=bins)
    if len(hist) == 0 or np.max(hist) == 0:
        return (points, None)
    best_bin_index = int(np.argmax(hist))
    band_center = int(round((edges[best_bin_index] + edges[best_bin_index + 1]) / 2.0))
    filtered = []
    for p in points:
        if abs(float(p.x) - float(band_center)) <= half_height:
            filtered.append(p)
    if len(filtered) < BAND_MIN_POINTS:
        return (points, {'band_center': band_center, 'half_height': half_height, 'used_fallback': True})
    return (filtered, {'band_center': band_center, 'half_height': half_height, 'used_fallback': False})

def fit_poly_from_points(points, degree):
    if points is None or len(points) < degree + 1:
        return None
    rows = np.array([p.x for p in points], dtype=np.float32)
    cols = np.array([p.y for p in points], dtype=np.float32)
    if len(np.unique(cols.astype(np.int32))) < degree + 1:
        return None
    try:
        return np.poly1d(np.polyfit(cols, rows, degree))
    except Exception:
        return None

def filter_points_by_poly_band(points, poly_line, above_px, below_px):
    if points is None or len(points) == 0 or poly_line is None:
        return []

    filtered = []

    for p in points:
        if hasattr(poly_line, 'is_col_allowed') and not poly_line.is_col_allowed(p.y):
            continue

        expected_row = float(poly_line(p.y))
        delta = float(p.x) - expected_row

        if -above_px <= delta <= below_px:
            filtered.append(p)

    return filtered

def point_set_metrics(points, image_shape):
    h, w = image_shape[:2]
    if points is None or len(points) == 0:
        return {'count': 0, 'width_frac': 0.0, 'height_frac': 1.0, 'continuity': 0.0, 'keep_score': -1000000000.0, 'row_span': 0.0, 'col_span': 0.0}
    rows = np.array([p.x for p in points], dtype=np.float32)
    cols = np.array([p.y for p in points], dtype=np.float32)
    col_min = float(np.min(cols))
    col_max = float(np.max(cols))
    row_min = float(np.min(rows))
    row_max = float(np.max(rows))
    col_span = max(1.0, col_max - col_min + 1.0)
    row_span = max(1.0, row_max - row_min + 1.0)
    width_frac = col_span / max(float(w), 1.0)
    height_frac = row_span / max(float(h), 1.0)
    active_cols = len(np.unique(cols.astype(np.int32)))
    continuity = active_cols / max(col_span, 1.0)
    score = 0.0
    score += 4.0 * min(width_frac / 0.55, 1.0)
    score += 2.0 * min(len(points) / 180.0, 1.0)
    score += 1.5 * min(continuity / 0.65, 1.0)
    score -= 2.0 * min(height_frac / 0.2, 1.0)
    return {'count': int(len(points)), 'width_frac': float(width_frac), 'height_frac': float(height_frac), 'continuity': float(continuity), 'keep_score': float(score), 'row_span': float(row_span), 'col_span': float(col_span)}

def estimate_selected_points_geometry(points, image_shape):
    h, w = image_shape[:2]
    if points is None or len(points) < 3:
        return {'row_span_frac': 0.0, 'slope': 0.0}
    rows = np.array([p.x for p in points], dtype=np.float32)
    cols = np.array([p.y for p in points], dtype=np.float32)
    row_span_frac = float(np.max(rows) - np.min(rows) + 1.0) / max(float(h), 1.0)
    if len(np.unique(cols.astype(np.int32))) >= 2:
        try:
            slope = float(np.polyfit(cols, rows, 1)[0])
        except Exception:
            slope = 0.0
    else:
        slope = 0.0
    return {'row_span_frac': float(row_span_frac), 'slope': float(slope)}

def smooth_series(values, window):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    window = max(1, int(window))
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode='valid')

def build_midline_points_from_vertical_columns(points, image_shape):
    h, w = image_shape[:2]
    if points is None or len(points) == 0:
        return []
    grouped = {}
    for p in points:
        col = int(round(p.y))
        row = float(p.x)
        if 0 <= col < w and 0 <= row < h:
            if col not in grouped:
                grouped[col] = []
            grouped[col].append(row)
    if len(grouped) < CURVED_BAND_MIN_MIDPOINT_COLUMNS:
        return []
    cols = np.array(sorted(grouped.keys()), dtype=np.float32)
    rows_mid = []
    for col in cols:
        values = np.array(grouped[int(col)], dtype=np.float32)
        rows_mid.append(float((np.min(values) + np.max(values)) / 2.0))
    rows_mid = np.array(rows_mid, dtype=np.float32)
    smooth_w = max(3, int(round(CURVED_BAND_MIDPOINT_SMOOTH_FRAC * max(float(w), 1.0))))
    rows_mid = smooth_series(rows_mid, smooth_w)
    mid_points = []
    for col, row in zip(cols, rows_mid):
        mid_points.append(Point(x=float(row), y=float(col)))
    return mid_points

def select_dense_midline_interval(mid_points, image_shape):
    h, w = image_shape[:2]
    if mid_points is None or len(mid_points) < CURVED_LOCAL_MIN_DENSE_COLUMNS:
        return []
    ordered = sorted(mid_points, key=lambda p: p.y)
    cols = np.array([p.y for p in ordered], dtype=np.float32)
    max_gap = max(3.0, CURVED_LOCAL_MAX_COLUMN_GAP_FRAC * max(float(w), 1.0))
    min_width = CURVED_LOCAL_MIN_DENSE_WIDTH_FRAC * max(float(w), 1.0)
    segments = []
    start = 0
    for i in range(1, len(ordered)):
        if cols[i] - cols[i - 1] > max_gap:
            segments.append((start, i))
            start = i
    segments.append((start, len(ordered)))
    best = None
    best_score = -1000000000.0
    for a, b in segments:
        segment = ordered[a:b]
        if len(segment) < CURVED_LOCAL_MIN_DENSE_COLUMNS:
            continue
        width = float(segment[-1].y - segment[0].y + 1.0)
        if width < min_width:
            continue
        rows = np.array([p.x for p in segment], dtype=np.float32)
        row_span_frac = float(np.max(rows) - np.min(rows) + 1.0) / max(float(h), 1.0)
        continuity = len(segment) / max(width, 1.0)
        score = 0.0
        score += 3.0 * min(width / max(0.55 * w, 1.0), 1.0)
        score += 2.0 * min(len(segment) / 120.0, 1.0)
        score += 1.5 * min(continuity / 0.55, 1.0)
        score -= 1.2 * max(0.0, row_span_frac - CURVED_LOCAL_MAX_AMPLITUDE_FRAC)
        if score > best_score:
            best_score = score
            best = segment
    if best is None:
        return []
    return best

def build_local_curve_model(mid_points, image_shape):
    dense_points = select_dense_midline_interval(mid_points, image_shape)
    if len(dense_points) < CURVED_LOCAL_MIN_DENSE_COLUMNS:
        return (None, {'valid': False, 'reason': 'no_dense_midline_interval', 'midpoints': int(0 if mid_points is None else len(mid_points)), 'dense_midpoints': int(len(dense_points))})
    h, w = image_shape[:2]
    cols = np.array([p.y for p in dense_points], dtype=np.float32)
    rows = np.array([p.x for p in dense_points], dtype=np.float32)
    smooth_w = max(3, int(round(CURVED_LOCAL_SMOOTH_FRAC * max(float(w), 1.0))))
    rows = smooth_series(rows, smooth_w)
    median_row = float(np.median(rows))
    max_jump = CURVED_LOCAL_EDGE_JUMP_FRAC * max(float(h), 1.0)
    rows = np.clip(rows, median_row - max_jump * 2.0, median_row + max_jump * 2.0)
    model = LocalCurveModel(cols=cols, rows=rows, image_shape=image_shape, extension_frac=CURVED_LOCAL_EXTENSION_FRAC, max_slope=CURVED_LOCAL_EXTENSION_MAX_SLOPE)
    ok, stable_info = validate_poly_stability(model, [Point(x=float(r), y=float(c)) for c, r in zip(cols, rows)], image_shape, CURVED_LOCAL_MAX_AMPLITUDE_FRAC, max_residual_frac=CURVED_LOCAL_MAX_RESIDUAL_FRAC, edge_jump_limit_frac=CURVED_LOCAL_EDGE_JUMP_FRAC)
    if not ok:
        return (model, {'valid': False, 'reason': stable_info.get('reason', 'local_curve_unstable'), 'stable_info': stable_info, 'midpoints': int(len(mid_points)), 'dense_midpoints': int(len(dense_points)), 'model_type': 'local_interpolated'})
    return (model, {'valid': True, 'reason': 'ok', 'stable_info': stable_info, 'midpoints': int(len(mid_points)), 'dense_midpoints': int(len(dense_points)), 'model_type': 'local_interpolated', 'col_range': [int(model.col_min), int(model.col_max)]})

def validate_poly_stability(poly_line, points, image_shape, max_amp_frac, max_residual_frac=None, edge_jump_limit_frac=None):
    h, w = image_shape[:2]
    if poly_line is None or points is None or len(points) == 0:
        return (False, {'reason': 'poly_absent'})
    if max_residual_frac is None:
        max_residual_frac = CURVED_BAND_MAX_RESIDUAL_FRAC
    if edge_jump_limit_frac is None:
        edge_jump_limit_frac = CURVED_BAND_REJECT_IF_EDGE_EXPLODES_FRAC
    cols = np.array([p.y for p in points], dtype=np.float32)
    rows = np.array([p.x for p in points], dtype=np.float32)
    c_min = int(max(0, np.min(cols)))
    c_max = int(min(w - 1, np.max(cols)))
    if c_max <= c_min:
        return (False, {'reason': 'invalid_col_range'})
    test_cols = np.linspace(c_min, c_max, num=80, dtype=np.float32)
    test_rows = poly_line(test_cols)
    poly_row_min = float(np.min(test_rows))
    poly_row_max = float(np.max(test_rows))
    poly_amplitude_frac = (poly_row_max - poly_row_min) / max(float(h), 1.0)
    residual = float(np.mean(np.abs(poly_line(cols) - rows)))
    residual_frac = residual / max(float(h), 1.0)
    if poly_amplitude_frac > max_amp_frac:
        return (False, {'reason': 'poly_amplitude_too_large', 'poly_amplitude_frac': float(poly_amplitude_frac), 'residual': float(residual), 'residual_frac': float(residual_frac), 'col_range': [int(c_min), int(c_max)]})
    if residual_frac > max_residual_frac:
        return (False, {'reason': 'poly_residual_too_large', 'poly_amplitude_frac': float(poly_amplitude_frac), 'residual': float(residual), 'residual_frac': float(residual_frac), 'col_range': [int(c_min), int(c_max)]})
    edge_cols = np.array([c_min, c_max], dtype=np.float32)
    edge_rows = poly_line(edge_cols)
    median_row = float(np.median(rows))
    edge_jump_frac = float(np.max(np.abs(edge_rows - median_row)) / max(float(h), 1.0))
    if edge_jump_frac > edge_jump_limit_frac and poly_amplitude_frac > 0.1:
        return (False, {'reason': 'poly_edges_unstable', 'poly_amplitude_frac': float(poly_amplitude_frac), 'edge_jump_frac': float(edge_jump_frac), 'residual': float(residual), 'residual_frac': float(residual_frac), 'col_range': [int(c_min), int(c_max)]})
    return (True, {'reason': 'ok', 'poly_amplitude_frac': float(poly_amplitude_frac), 'edge_jump_frac': float(edge_jump_frac), 'residual': float(residual), 'residual_frac': float(residual_frac), 'col_range': [int(c_min), int(c_max)]})

def build_curved_band_candidate(points, image_shape):
    if points is None or len(points) < CURVED_BAND_MIN_POINTS:
        return ([], None, {'valid': False, 'reason': 'too_few_points'})
    mid_points = build_midline_points_from_vertical_columns(points, image_shape)
    if len(mid_points) < CURVED_BAND_MIN_MIDPOINT_COLUMNS:
        return ([], None, {'valid': False, 'reason': 'too_few_midpoints', 'midpoints': int(len(mid_points))})
    candidates = []
    if CURVED_LOCAL_ENABLE:
        local_model, local_info = build_local_curve_model(mid_points, image_shape)
        if local_model is not None:
            local_points = filter_points_by_poly_band(points, local_model, CURVED_BAND_ABOVE_PX, CURVED_BAND_BELOW_PX)
            local_metrics = point_set_metrics(local_points, image_shape)
            local_valid = local_info.get('valid', False) and len(local_points) >= CURVED_BAND_MIN_KEEP_POINTS and (local_metrics['width_frac'] >= CURVED_MIN_WIDTH_FRAC) and (local_metrics['continuity'] >= CURVED_ACCEPT_MIN_CONTINUITY) and (local_metrics['height_frac'] <= CURVED_ACCEPT_MAX_HEIGHT_FRAC)
            local_info = dict(local_info)
            local_info.update({'valid': bool(local_valid), 'reason': 'ok' if local_valid else local_info.get('reason', 'weak_local_curve_candidate'), 'metrics': local_metrics, 'degree': 0, 'half_height': int(max(CURVED_BAND_ABOVE_PX, CURVED_BAND_BELOW_PX)), 'input_points': int(len(points)), 'kept_points': int(len(local_points)), 'model_type': 'local_interpolated'})
            candidates.append((local_points, local_model, local_info))
    if CURVED_ALLOW_POLYNOMIAL_BACKUP:
        degree = min(CURVED_BAND_POLY_DEGREE, max(1, len(mid_points) - 1))
        curved_poly = fit_poly_from_points(mid_points, degree)

        if curved_poly is not None:
            ok, stable_info = validate_poly_stability(curved_poly, mid_points, image_shape, CURVED_BAND_MAX_POLY_AMPLITUDE_FRAC)
            poly_points = filter_points_by_poly_band(points, curved_poly, CURVED_BAND_ABOVE_PX, CURVED_BAND_BELOW_PX) if ok else []
            poly_metrics = point_set_metrics(poly_points, image_shape)
            poly_valid = ok and len(poly_points) >= CURVED_BAND_MIN_KEEP_POINTS and (poly_metrics['width_frac'] >= CURVED_MIN_WIDTH_FRAC) and (poly_metrics['continuity'] >= CURVED_ACCEPT_MIN_CONTINUITY) and (poly_metrics['height_frac'] <= CURVED_ACCEPT_MAX_HEIGHT_FRAC)
            poly_info = {'valid': bool(poly_valid), 'reason': 'ok' if poly_valid else stable_info.get('reason', 'weak_poly_curve_candidate'), 'metrics': poly_metrics, 'stable_info': stable_info, 'degree': int(degree), 'half_height': int(max(CURVED_BAND_ABOVE_PX, CURVED_BAND_BELOW_PX)), 'input_points': int(len(points)), 'midpoints': int(len(mid_points)), 'kept_points': int(len(poly_points)), 'model_type': 'polynomial_backup'}
            candidates.append((poly_points, curved_poly, poly_info))
    valid_candidates = [item for item in candidates if item[2].get('valid', False)]
    if len(valid_candidates) == 0:
        if len(candidates) > 0:
            best_failed = max(candidates, key=lambda item: item[2].get('metrics', {}).get('keep_score', -1000000000.0))
            return (best_failed[0], best_failed[1], best_failed[2])
        return ([], None, {'valid': False, 'reason': 'no_curve_candidate', 'midpoints': int(len(mid_points))})
    best = max(valid_candidates, key=lambda item: item[2]['metrics']['keep_score'])
    return (best[0], best[1], best[2])

def get_horizontal_band_center_from_info(band_info):
    if not isinstance(band_info, dict):
        return None

    info = band_info

    if band_info.get('mode') == 'horizontal':
        info = band_info.get('horizontal_info')

    if isinstance(info, dict) and 'band_center' in info:
        return float(info['band_center'])

    return None


def constant_row_poly(row):
    return np.poly1d([0.0, float(row)])


def safe_initial_poly_from_band_points(band_points, image_shape, band_mode, band_info, band_poly):
    if band_points is None or len(band_points) == 0:
        return None, []

    if band_poly is not None:
        max_amp = INITIAL_POLY_MAX_AMPLITUDE_FRAC_LINEAR if band_mode == 'linear' else INITIAL_POLY_MAX_AMPLITUDE_FRAC
        ok, _ = validate_poly_stability(
            band_poly,
            band_points,
            image_shape,
            max_amp,
            max_residual_frac=INITIAL_POLY_MAX_RESIDUAL_FRAC
        )

        if ok or band_mode == 'linear':
            return band_poly, band_points

    rows = np.array([p.x for p in band_points], dtype=np.float32)
    cols = np.array([p.y for p in band_points], dtype=np.float32)

    if band_mode == 'horizontal' and INITIAL_POLY_HORIZONTAL_USE_BAND_CENTER:
        band_center = get_horizontal_band_center_from_info(band_info)

        if band_center is None:
            band_center = float(np.median(rows))

        return constant_row_poly(band_center), band_points

    wanted_degree = FINAL_POLY_DEGREE_CURVED if band_mode == 'curved' else FINAL_POLY_DEGREE_HORIZONTAL
    wanted_degree = min(max(1, wanted_degree), len(band_points) - 1)

    if len(np.unique(cols.astype(np.int32))) >= wanted_degree + 1:
        poly_line, _ = IdnetifyPoly(rows.tolist(), cols.tolist(), wanted_degree)
        max_amp = INITIAL_POLY_MAX_AMPLITUDE_FRAC_LINEAR if band_mode == 'linear' else INITIAL_POLY_MAX_AMPLITUDE_FRAC
        ok, _ = validate_poly_stability(
            poly_line,
            band_points,
            image_shape,
            max_amp,
            max_residual_frac=INITIAL_POLY_MAX_RESIDUAL_FRAC
        )

        if ok:
            filtered_points = Fit(band_points, poly_line, POLY_DEVIATION)

            if len(filtered_points) >= max(3, min(POLY_MIN_POINTS, len(band_points))):
                return poly_line, filtered_points

    return constant_row_poly(float(np.median(rows))), band_points


def safe_final_poly_from_points(points, image_shape, band_mode):
    if points is None or len(points) < 2:
        return (None, {'valid': False, 'reason': 'too_few_points'})
    wanted_degree = FINAL_POLY_DEGREE_CURVED if band_mode == 'curved' else FINAL_POLY_DEGREE_HORIZONTAL
    wanted_degree = min(wanted_degree, len(points) - 1)
    wanted_degree = max(1, wanted_degree)
    rows = [p.x for p in points]
    cols = [p.y for p in points]
    poly_line, _ = IdnetifyPoly(rows, cols, wanted_degree)
    max_amp = FINAL_POLY_MAX_AMPLITUDE_FRAC_LINEAR if band_mode == 'linear' else FINAL_POLY_MAX_AMPLITUDE_FRAC
    ok, info = validate_poly_stability(poly_line, points, image_shape, max_amp)
    if ok:
        info['valid'] = True
        info['degree'] = int(wanted_degree)
        return (poly_line, info)
    if wanted_degree != 1:
        poly_linear, _ = IdnetifyPoly(rows, cols, 1)
        ok_linear, linear_info = validate_poly_stability(poly_linear, points, image_shape, max_amp)
        linear_info['valid'] = bool(ok_linear)
        linear_info['degree'] = 1
        linear_info['fallback_from_degree'] = int(wanted_degree)
        if ok_linear:
            return (poly_linear, linear_info)
    rows_np = np.array(rows, dtype=np.float32)
    median_row = float(np.median(rows_np))
    constant_poly = np.poly1d([0.0, median_row])
    constant_info = {'valid': True, 'reason': 'constant_line_fallback_after_unstable_poly', 'degree': 1, 'fallback_from_degree': int(wanted_degree)}
    return (constant_poly, constant_info)


def row_median_near_column(points, center_col, window_px):
    if points is None or len(points) == 0:
        return None

    rows = []

    for p in points:
        if abs(float(p.y) - float(center_col)) <= float(window_px):
            rows.append(float(p.x))

    if len(rows) == 0:
        return None

    return float(np.median(np.array(rows, dtype=np.float32)))


def make_line_from_two_samples(c1, r1, c2, r2):
    if c1 is None or c2 is None or r1 is None or r2 is None:
        return None

    if abs(float(c2) - float(c1)) < 1.0:
        return None

    slope = (float(r2) - float(r1)) / (float(c2) - float(c1))
    intercept = float(r1) - slope * float(c1)

    return np.poly1d([float(slope), float(intercept)])


def refine_line_candidate(points, initial_poly, image_shape):
    if points is None or len(points) < LINEAR_BAND_MIN_POINTS or initial_poly is None:
        return [], None, {
            'valid': False,
            'reason': 'too_few_points_or_missing_line'
        }

    current_points = list(points)
    best_poly = initial_poly
    best_points = filter_points_by_poly_band(
        points,
        best_poly,
        LINEAR_BAND_ABOVE_PX,
        LINEAR_BAND_BELOW_PX
    )

    for _ in range(max(1, int(LINEAR_BAND_MAX_ITER))):
        if current_points is None or len(current_points) < LINEAR_BAND_MIN_POINTS:
            break

        rows = np.array([p.x for p in current_points], dtype=np.float32)
        cols = np.array([p.y for p in current_points], dtype=np.float32)

        if len(np.unique(cols.astype(np.int32))) < 2:
            break

        try:
            poly = np.poly1d(np.polyfit(cols, rows, 1))
        except Exception:
            break

        filtered = filter_points_by_poly_band(
            points,
            poly,
            LINEAR_BAND_ABOVE_PX,
            LINEAR_BAND_BELOW_PX
        )

        if len(filtered) < LINEAR_BAND_MIN_POINTS:
            break

        best_poly = poly
        best_points = filtered
        current_points = filtered

    metrics = point_set_metrics(best_points, image_shape)

    if best_poly is None:
        slope = 999.0
    else:
        coeffs = np.asarray(best_poly.coeffs, dtype=np.float32)
        slope = float(coeffs[0]) if len(coeffs) >= 2 else 0.0

    valid = (
        len(best_points) >= LINEAR_BAND_MIN_POINTS
        and metrics['width_frac'] >= LINEAR_BAND_MIN_WIDTH_FRAC
        and metrics['continuity'] >= LINEAR_ACCEPT_MIN_CONTINUITY
        and metrics['height_frac'] <= LINEAR_BAND_MAX_HEIGHT_FRAC
        and abs(slope) <= LINEAR_BAND_MAX_SLOPE_ABS
    )

    info = {
        'valid': bool(valid),
        'reason': 'ok' if valid else 'weak_linear_candidate',
        'metrics': metrics,
        'degree': 1,
        'half_height': int(max(LINEAR_BAND_ABOVE_PX, LINEAR_BAND_BELOW_PX)),
        'input_points': int(len(points)),
        'kept_points': int(len(best_points)),
        'slope': float(slope),
        'model_type': 'linear_fallback'
    }

    return best_points, best_poly, info


def build_linear_band_candidate(points, image_shape):
    if not LINEAR_BAND_ENABLE:
        return [], None, {
            'valid': False,
            'reason': 'linear_disabled'
        }

    if points is None or len(points) < LINEAR_BAND_MIN_POINTS:
        return [], None, {
            'valid': False,
            'reason': 'too_few_points'
        }

    h, w = image_shape[:2]
    rows = np.array([p.x for p in points], dtype=np.float32)
    cols = np.array([p.y for p in points], dtype=np.float32)

    if len(np.unique(cols.astype(np.int32))) < 2:
        return [], None, {
            'valid': False,
            'reason': 'too_few_unique_columns'
        }

    candidates = []

    try:
        candidates.append(np.poly1d(np.polyfit(cols, rows, 1)))
    except Exception:
        pass

    # Candidati robusti din percentile: evitam ca un grup mic de puncte sa traga linia.
    q_pairs = [(5, 95), (10, 90), (15, 85), (20, 80)]
    window_px = max(4, int(round(0.04 * max(float(w), 1.0))))

    for q1, q2 in q_pairs:
        c1 = float(np.percentile(cols, q1))
        c2 = float(np.percentile(cols, q2))
        r1 = row_median_near_column(points, c1, window_px)
        r2 = row_median_near_column(points, c2, window_px)
        line = make_line_from_two_samples(c1, r1, c2, r2)

        if line is not None:
            candidates.append(line)

    if len(candidates) == 0:
        return [], None, {
            'valid': False,
            'reason': 'line_fit_failed'
        }

    evaluated = []

    for candidate in candidates:
        pts, poly, info = refine_line_candidate(points, candidate, image_shape)
        metrics = info.get('metrics', point_set_metrics(pts, image_shape))
        keep_ratio = len(pts) / max(len(points), 1)
        slope = abs(float(info.get('slope', 999.0)))

        score = 0.0
        score += metrics['keep_score']
        score += 1.2 * min(metrics['width_frac'] / 0.55, 1.0)
        score += 0.9 * min(keep_ratio / 0.55, 1.0)
        score -= 0.6 * min(slope / 0.75, 1.0)

        info = dict(info)
        info['keep_ratio'] = float(keep_ratio)
        info['selection_score'] = float(score)
        evaluated.append((score, pts, poly, info))

    best_score, best_points, best_poly, best_info = max(evaluated, key=lambda item: item[0])

    best_info['selection_score'] = float(best_score)

    return best_points, best_poly, best_info


def choose_horizontal_or_curved_band(selected_points, image_shape):
    horizontal_points, horizontal_raw_info = filter_travelers_by_main_horizontal_band(selected_points, image_shape)
    curved_points, curved_poly, curved_info = build_curved_band_candidate(selected_points, image_shape)
    linear_points, linear_poly, linear_info = build_linear_band_candidate(selected_points, image_shape)

    h_metrics = point_set_metrics(horizontal_points, image_shape)
    c_metrics = point_set_metrics(curved_points, image_shape)
    l_metrics = point_set_metrics(linear_points, image_shape)

    total_points = max(len(selected_points), 1)
    horizontal_keep_ratio = len(horizontal_points) / total_points
    curved_keep_ratio = len(curved_points) / total_points
    linear_keep_ratio = len(linear_points) / total_points

    geometry = estimate_selected_points_geometry(selected_points, image_shape)

    horizontal_good = (
        horizontal_keep_ratio >= HORIZONTAL_GOOD_KEEP_RATIO
        and h_metrics['width_frac'] >= HORIZONTAL_GOOD_WIDTH_FRAC
        and h_metrics['height_frac'] <= CURVED_ACCEPT_MAX_HEIGHT_FRAC
    )

    force_non_horizontal = (
        geometry['row_span_frac'] >= FORCE_CURVED_ROW_SPAN_FRAC
        or abs(geometry['slope']) >= FORCE_CURVED_SLOPE_ABS
    )

    curve_shape_ok = (
        curved_info.get('valid', False)
        and curved_keep_ratio >= CURVED_ACCEPT_MIN_KEEP_RATIO
        and c_metrics['continuity'] >= CURVED_ACCEPT_MIN_CONTINUITY
        and c_metrics['height_frac'] <= CURVED_ACCEPT_MAX_HEIGHT_FRAC
    )

    linear_shape_ok = (
        linear_info.get('valid', False)
        and linear_keep_ratio >= LINEAR_ACCEPT_MIN_KEEP_RATIO
        and l_metrics['continuity'] >= LINEAR_ACCEPT_MIN_CONTINUITY
        and l_metrics['height_frac'] <= LINEAR_BAND_MAX_HEIGHT_FRAC
    )

    curved_horizontal_loses_width = c_metrics['width_frac'] - h_metrics['width_frac'] >= CURVED_ACCEPT_IF_HORIZONTAL_WIDTH_LOSS_FRAC
    curved_horizontal_loses_points = curved_keep_ratio - horizontal_keep_ratio >= CURVED_ACCEPT_IF_HORIZONTAL_KEEP_LOSS

    linear_horizontal_loses_width = l_metrics['width_frac'] - h_metrics['width_frac'] >= LINEAR_ACCEPT_IF_HORIZONTAL_WIDTH_LOSS_FRAC
    linear_horizontal_loses_points = linear_keep_ratio - horizontal_keep_ratio >= LINEAR_ACCEPT_IF_HORIZONTAL_KEEP_LOSS

    curved_better = (
        curve_shape_ok
        and c_metrics['keep_score'] >= h_metrics['keep_score'] * CURVED_MIN_SCORE_GAIN
    )

    curved_rescue = (
        curve_shape_ok
        and force_non_horizontal
        and not horizontal_good
        and (curved_horizontal_loses_width or curved_horizontal_loses_points)
        and c_metrics['keep_score'] >= h_metrics['keep_score'] * 0.92
    )

    horizontal_clearly_bad_for_linear = (
        horizontal_keep_ratio < 0.34
        or h_metrics['width_frac'] < 0.28
    )

    linear_better = (
        linear_shape_ok
        and force_non_horizontal
        and horizontal_clearly_bad_for_linear
        and (linear_horizontal_loses_width or linear_horizontal_loses_points)
        and l_metrics['keep_score'] >= h_metrics['keep_score'] * LINEAR_ACCEPT_SCORE_GAIN
    )

    # Daca si curba locala si linia sunt valide, alegem varianta mai stabila.
    # Linia are prioritate cand curba nu aduce clar mai multa acoperire, fiindca evita arcurile.
    if curved_better or curved_rescue:
        curved_gain_over_linear = c_metrics['keep_score'] - l_metrics['keep_score']
        curved_width_gain = c_metrics['width_frac'] - l_metrics['width_frac']

        if linear_better and curved_gain_over_linear < 0.35 and curved_width_gain < 0.08:
            info = {
                'mode': 'linear',
                'type': 'linear_fallback',
                'reason': 'linear_preferred_over_small_curve_gain',
                'used_fallback': False,
                'degree': 1,
                'half_height': max(LINEAR_BAND_ABOVE_PX, LINEAR_BAND_BELOW_PX),
                'horizontal_keep_ratio': float(horizontal_keep_ratio),
                'curved_keep_ratio': float(curved_keep_ratio),
                'linear_keep_ratio': float(linear_keep_ratio),
                'horizontal_metrics': h_metrics,
                'curved_metrics': c_metrics,
                'linear_metrics': l_metrics,
                'curved_info': curved_info,
                'linear_info': linear_info,
                'geometry': geometry,
                'poly_line': linear_poly
            }
            return 'linear', linear_points, info, linear_poly

        reason = 'curved_better' if curved_better else 'curved_rescue_horizontal_lost_contour'
        info = {
            'mode': 'curved',
            'type': curved_info.get('model_type', 'curved'),
            'reason': reason,
            'used_fallback': False,
            'degree': curved_info.get('degree', CURVED_BAND_POLY_DEGREE),
            'half_height': max(CURVED_BAND_ABOVE_PX, CURVED_BAND_BELOW_PX),
            'horizontal_keep_ratio': float(horizontal_keep_ratio),
            'curved_keep_ratio': float(curved_keep_ratio),
            'linear_keep_ratio': float(linear_keep_ratio),
            'horizontal_metrics': h_metrics,
            'curved_metrics': c_metrics,
            'linear_metrics': l_metrics,
            'curved_info': curved_info,
            'linear_info': linear_info,
            'geometry': geometry,
            'poly_line': curved_poly
        }
        return 'curved', curved_points, info, curved_poly

    if linear_better:
        info = {
            'mode': 'linear',
            'type': 'linear_fallback',
            'reason': 'linear_rescue_horizontal_lost_oblique_contour',
            'used_fallback': False,
            'degree': 1,
            'half_height': max(LINEAR_BAND_ABOVE_PX, LINEAR_BAND_BELOW_PX),
            'horizontal_keep_ratio': float(horizontal_keep_ratio),
            'curved_keep_ratio': float(curved_keep_ratio),
            'linear_keep_ratio': float(linear_keep_ratio),
            'horizontal_metrics': h_metrics,
            'curved_metrics': c_metrics,
            'linear_metrics': l_metrics,
            'curved_info': curved_info,
            'linear_info': linear_info,
            'geometry': geometry,
            'poly_line': linear_poly
        }
        return 'linear', linear_points, info, linear_poly

    reason = 'horizontal_stable'

    if not curved_info.get('valid', False) and not linear_info.get('valid', False):
        reason = 'horizontal_curved_and_linear_rejected'
    elif horizontal_good:
        reason = 'horizontal_good_enough'
    elif not curve_shape_ok and not linear_shape_ok:
        reason = 'horizontal_no_safe_non_horizontal_candidate'
    elif not (curved_horizontal_loses_width or curved_horizontal_loses_points or linear_horizontal_loses_width or linear_horizontal_loses_points):
        reason = 'horizontal_non_horizontal_gain_too_small'

    info = {
        'mode': 'horizontal',
        'type': 'horizontal_or_linear_fallback',
        'reason': reason,
        'horizontal_info': horizontal_raw_info,
        'horizontal_keep_ratio': float(horizontal_keep_ratio),
        'curved_keep_ratio': float(curved_keep_ratio),
        'linear_keep_ratio': float(linear_keep_ratio),
        'horizontal_metrics': h_metrics,
        'curved_metrics': c_metrics,
        'linear_metrics': l_metrics,
        'curved_info': curved_info,
        'linear_info': linear_info,
        'geometry': geometry
    }

    return 'horizontal', horizontal_points, info, None

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
    horizontal_info = band_info
    if isinstance(band_info, dict) and band_info.get('mode') == 'horizontal':
        horizontal_info = band_info.get('horizontal_info')
    if isinstance(horizontal_info, dict) and 'band_center' in horizontal_info:
        h, w = result.shape[:2]
        center = int(horizontal_info['band_center'])
        half_height = int(horizontal_info['half_height'])
        y1 = max(0, center - half_height)
        y2 = min(h - 1, center + half_height)
        cv2.line(result, (0, center), (w - 1, center), (0, 255, 255), 1)
        cv2.line(result, (0, y1), (w - 1, y1), (0, 165, 255), 1)
        cv2.line(result, (0, y2), (w - 1, y2), (0, 165, 255), 1)
    if isinstance(band_info, dict) and band_info.get('mode') in ['curved', 'linear']:
        poly_line = band_info.get('poly_line')
        if poly_line is not None:
            h, w = result.shape[:2]
            center_points = []
            top_points = []
            bottom_points = []
            if band_info.get('mode') == 'linear':
                half_above = int(LINEAR_BAND_ABOVE_PX)
                half_below = int(LINEAR_BAND_BELOW_PX)
            else:
                half_above = int(CURVED_BAND_ABOVE_PX)
                half_below = int(CURVED_BAND_BELOW_PX)
            if hasattr(poly_line, 'active_col_min') and hasattr(poly_line, 'active_col_max'):
                col_start = max(0, int(np.floor(poly_line.active_col_min)))
                col_end = min(w - 1, int(np.ceil(poly_line.active_col_max)))
            else:
                col_start = 0
                col_end = w - 1

            for col in range(col_start, col_end + 1):
                row_center = int(round(poly_line(col)))
                row_top = row_center - half_above
                row_bottom = row_center + half_below

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

def IdnetifyPoly(X_, Y_, order):
    if len(X_) < order + 1 or len(Y_) < order + 1:
        return (None, [])
    poly_line = np.poly1d(np.polyfit(Y_, X_, order))
    fitted_X = poly_line(Y_)
    return (poly_line, fitted_X)

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
        return (working, CropBox(0, 0, h, w, False), working.copy())
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
    roi_box = CropBox(top=max(0, top), left=max(0, left), bottom=min(h, bottom), right=min(w, right), valid=True)
    roi_box = clamp_box(roi_box, binary_mask.shape)
    if not is_principal_roi_valid(roi_box, binary_mask.shape):
        roi_box.valid = False
    roi = working[roi_box.top:roi_box.bottom, roi_box.left:roi_box.right].copy()
    return (working, roi_box, roi)

def draw_principal_roi_overlay(crop_gray, roi_box, filtered_points):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()
    if roi_box is not None and roi_box.valid:
        cv2.rectangle(result, (roi_box.left, roi_box.top), (roi_box.right - 1, roi_box.bottom - 1), (0, 255, 255), 2)
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
        return (full_mask, full_contour_mask, None)
    if roi_box is None or not roi_box.valid:
        return (full_mask, full_contour_mask, None)
    if principal_roi.ndim == 3:
        principal_roi = cv2.cvtColor(principal_roi, cv2.COLOR_BGR2GRAY)
    binary = (principal_roi > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return (full_mask, full_contour_mask, None)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (full_mask, full_contour_mask, None)
    min_area = max(3, int(round(PRINCIPAL_COMPONENT_MIN_AREA_FRAC * full_shape[0] * full_shape[1])))
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
        return (full_mask, full_contour_mask, None)
    component_roi = np.zeros_like(binary, dtype=np.uint8)
    component_roi[labels == best_label] = 255
    kernel_size = max(1, int(PRINCIPAL_COMPONENT_DILATE_KERNEL))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    component_roi = cv2.dilate(component_roi, kernel, iterations=1)
    contours = get_contours(component_roi)
    if len(contours) == 0:
        return (full_mask, full_contour_mask, None)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_full = largest_contour.copy()
    contour_full[:, 0, 0] += roi_box.left
    contour_full[:, 0, 1] += roi_box.top
    full_mask[roi_box.top:roi_box.bottom, roi_box.left:roi_box.right] = component_roi[:roi_box.height, :roi_box.width]
    cv2.drawContours(full_contour_mask, [contour_full], -1, 255, 1)
    return (full_mask, full_contour_mask, contour_full)

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
        return ([], [], [])
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
    return (filtered_points, X_, Y_)

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
        return (filtered_mask, filtered_contour_mask, None, None, [])
    filtered_points, X_, Y_ = Fit2(contour_points, poly_line, PRINCIPAL_CONTOUR_DEVIATION)
    filtered_contour = points_to_contour(filtered_points)
    if filtered_contour is None or len(filtered_points) == 0:
        return (filtered_mask, filtered_contour_mask, None, None, [])
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
    return (filtered_mask, filtered_contour_mask, filtered_contour, final_poly, filtered_points)

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
        return (invalid, invalid)
    cols = np.array([p.y for p in principal_points], dtype=np.float32)
    rows = np.array([p.x for p in principal_points], dtype=np.float32)
    if len(cols) == 0 or len(rows) == 0:
        return (invalid, invalid)
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
    left_box = CropBox(top=max(0, left_top), left=0, bottom=h, right=left_right, valid=True)
    right_box = CropBox(top=max(0, right_top), left=right_left, bottom=h, right=w, valid=True)
    left_box = clamp_box(left_box, image_shape)
    right_box = clamp_box(right_box, image_shape)
    if not is_box_valid(left_box, image_shape):
        left_box.valid = False
    if not is_box_valid(right_box, image_shape):
        right_box.valid = False
    return (left_box, right_box)

def crop_box_region(image, box):
    if box is None or not box.valid:
        return np.zeros((1, 1), dtype=np.uint8)
    return image[box.top:box.bottom, box.left:box.right].copy()

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
        return (full_mask, full_contour_mask, None, local_debug)
    if roi_binary_override is not None:
        roi_binary = roi_binary_override.copy()
    else:
        roi_binary = crop_box_region(binary_mask, roi_box)
    if roi_binary is None or roi_binary.size == 0:
        return (full_mask, full_contour_mask, None, local_debug)
    local_points = []
    ExtractContour(roi_binary, local_points)
    if len(local_points) < SECONDARY_MIN_POINTS:
        local_debug = draw_traveler_points(roi_binary, local_points)
        return (full_mask, full_contour_mask, None, local_debug)
    distances = removeOutliers(local_points, OUTLIER_DISTANCE)
    selected_points, _, _, scored_components = ExtractBestPleuralTravelerComponent(distances, local_points, roi_binary.shape)
    band_points, band_info = filter_travelers_by_main_horizontal_band(selected_points, roi_binary.shape)
    X_ = [p.x for p in band_points]
    Y_ = [p.y for p in band_points]
    if len(band_points) >= POLY_MIN_POINTS:
        local_poly, _ = IdnetifyPoly(X_, Y_, POLY_DEGREE)
        filtered_points = Fit(band_points, local_poly, POLY_DEVIATION)
    else:
        local_poly = None
        filtered_points = band_points
    local_debug = draw_poly_filtered_travelers(roi_binary, band_points, filtered_points, local_poly)
    if len(filtered_points) < SECONDARY_MIN_POINTS:
        return (full_mask, full_contour_mask, None, local_debug)
    _, local_roi_box, local_roi = build_principal_roi_from_travelers(roi_binary, filtered_points)
    local_component_mask, local_component_contour_mask, local_contour = extract_largest_component_contour_from_roi(local_roi, local_roi_box, roi_binary.shape)
    if local_contour is None:
        return (full_mask, full_contour_mask, None, local_debug)
    global_contour = contour_local_to_global(local_contour, roi_box)
    if global_contour is None:
        return (full_mask, full_contour_mask, None, local_debug)
    if lateral_poly is not None:
        global_points = contour_to_points(global_contour)
        filtered_global_points, _, _ = Fit2(global_points, lateral_poly, SECONDARY_CONTOUR_DEVIATION)
        filtered_global_contour = points_to_contour(filtered_global_points)
    else:
        filtered_global_contour = global_contour
    if filtered_global_contour is None or len(filtered_global_contour) < 3:
        return (full_mask, full_contour_mask, None, local_debug)
    cv2.drawContours(full_mask, [filtered_global_contour], -1, 255, -1)
    cv2.drawContours(full_contour_mask, [filtered_global_contour], -1, 255, 1)
    return (full_mask, full_contour_mask, filtered_global_contour, local_debug)

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
    if side == 'left':
        order = np.argsort(pts[:, 0])
    else:
        order = np.argsort(-pts[:, 0])
    sample_count = max(1, min(sample_count, len(order)))
    selected = pts[order[:sample_count]]
    return selected.astype(np.int32)

def find_closest_points(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return (None, None)
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
    return (best_a, best_b)

def connect_two_contours(mask, contour_a, side_a, contour_b, side_b):
    points_a = get_endpoint_candidates(contour_a, side_a)
    points_b = get_endpoint_candidates(contour_b, side_b)
    p1, p2 = find_closest_points(points_a, points_b)
    if p1 is None or p2 is None:
        return mask
    cv2.line(mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, MERGE_CONNECTION_THICKNESS, cv2.LINE_AA)
    hull_points = np.vstack([points_a, points_b]).astype(np.int32)
    if len(hull_points) >= 3:
        hull = cv2.convexHull(hull_points.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 255, -1)
    return mask

def merge_pleural_components(shape, principal_contour, left_contour, right_contour):
    merged_mask = np.zeros(shape[:2], dtype=np.uint8)
    merged_contour_mask = np.zeros(shape[:2], dtype=np.uint8)
    if not contour_is_valid(principal_contour, min_area=1):
        return (merged_mask, merged_contour_mask, None)
    cv2.drawContours(merged_mask, [principal_contour], -1, 255, -1)
    if contour_is_valid(left_contour):
        cv2.drawContours(merged_mask, [left_contour], -1, 255, -1)
        merged_mask = connect_two_contours(merged_mask, left_contour, 'right', principal_contour, 'left')
    if contour_is_valid(right_contour):
        cv2.drawContours(merged_mask, [right_contour], -1, 255, -1)
        merged_mask = connect_two_contours(merged_mask, principal_contour, 'right', right_contour, 'left')
    kernel = np.ones((3, 3), dtype=np.uint8)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours = get_contours(merged_mask)
    if len(contours) == 0:
        return (merged_mask, merged_contour_mask, None)
    final_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(merged_contour_mask, [final_contour], -1, 255, 1)
    return (merged_mask, merged_contour_mask, final_contour)

def draw_final_merged_overlay(crop_gray, merged_contour_mask):
    if crop_gray.ndim == 2:
        result = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = crop_gray.copy()
    if merged_contour_mask is not None:
        result[merged_contour_mask > 0] = (0, 255, 0)
    return result

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

def limit_pleura_thickness_by_poly(mask, poly_line):
    limited_mask = np.zeros_like(mask, dtype=np.uint8)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    if mask is None or mask.size == 0:
        return (limited_mask, contour_mask, None)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if poly_line is None:
        contours = get_contours(mask)
        if len(contours) == 0:
            return (mask.copy(), contour_mask, None)
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_mask, [contour], -1, 255, 1)
        return (mask.copy(), contour_mask, contour)
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
            return (limited_mask, contour_mask, None)
        contour = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(limited_mask, dtype=np.uint8)
        cv2.drawContours(clean, [contour], -1, 255, -1)
        limited_mask = clean
    else:
        contours = get_contours(limited_mask)
        if len(contours) == 0:
            return (limited_mask, contour_mask, None)
        contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(contour_mask, [contour], -1, 255, 1)
    return (limited_mask, contour_mask, contour)

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

def save_debug_image(index, step_number, name, image):
    ensure_dir(DEBUG_IMPORTANT_STEPS_DIR)
    if image is None:
        return
    base = f'{index:02d}_{step_number:02d}_{name}.png'
    path = os.path.join(DEBUG_IMPORTANT_STEPS_DIR, base)
    cv2.imwrite(path, image)


def save_important_identification_steps(index, crop, palette, binary, traveler_points, selected_points, band_points, band_info, filtered_points, poly_line, roi_box, principal_component_contour, filtered_principal_contour, final_poly, left_secondary_contour, right_secondary_contour, merged_pleura_contour, limited_pleura_contour):
    save_debug_image(index, 1, 'crop', crop)
    save_debug_image(index, 2, 'palette_7', palette)
    save_debug_image(index, 3, 'binary_mask', binary)
    save_debug_image(index, 4, 'traveler_points', draw_traveler_points(crop, traveler_points))
    save_debug_image(index, 5, 'selected_component', draw_selected_pleural_travelers(crop, traveler_points, selected_points))
    save_debug_image(index, 6, 'band_filtered', draw_band_filtered_travelers(crop, selected_points, band_points, band_info))
    save_debug_image(index, 7, 'poly_filtered', draw_poly_filtered_travelers(crop, band_points, filtered_points, poly_line))
    save_debug_image(index, 8, 'principal_roi', draw_principal_roi_overlay(crop, roi_box, filtered_points))
    save_debug_image(index, 9, 'principal_component', draw_principal_component_overlay(crop, principal_component_contour))
    save_debug_image(index, 10, 'filtered_principal', draw_filtered_principal_overlay(crop, filtered_principal_contour, final_poly))
    save_debug_image(index, 11, 'secondary_components', draw_secondary_component_overlay(crop, filtered_principal_contour, left_secondary_contour, right_secondary_contour))
    save_debug_image(index, 12, 'merged_pleura', draw_final_merged_overlay(crop, merged_pleura_contour))
    save_debug_image(index, 13, 'limited_final', draw_limited_pleura_overlay(crop, limited_pleura_contour, final_poly))

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
            cv2.drawContours(result, [contour_on_original], -1, (0, 255, 0), FINAL_CONTOUR_THICKNESS)
    base = f'{index:02d}'
    output_path = os.path.join(FINAL_CONTOUR_ON_ORIGINAL_DIR, base + '_final_contour_on_original.png')
    cv2.imwrite(output_path, result)

def save_crop(index):
    path = find_image_path(index)
    if path is None:
        return False
    image_bgr = read_image_bgr(path)
    crop, crop_box = crop_border(image_bgr)
    palette = reduce_palette_7(crop)
    binary, _ = binarize_palette_7(palette)
    traveler_points = []
    ExtractContour(binary, traveler_points)
    distances = removeOutliers(traveler_points, OUTLIER_DISTANCE)
    selected_points, _, _, _ = ExtractBestPleuralTravelerComponent(distances, traveler_points, binary.shape)
    band_mode, band_points, band_info, band_poly = choose_horizontal_or_curved_band(selected_points, binary.shape)
    poly_line, filtered_points = safe_initial_poly_from_band_points(
        band_points,
        binary.shape,
        band_mode,
        band_info,
        band_poly
    )
    _, roi_box, principal_roi = build_principal_roi_from_travelers(binary, filtered_points)
    _, principal_component_contour, principal_contour = extract_largest_component_contour_from_roi(principal_roi, roi_box, binary.shape)
    _, filtered_principal_contour, filtered_principal_ctr, final_poly, final_points = build_filtered_principal_component(principal_contour, poly_line, binary.shape)
    safe_final_poly, _ = safe_final_poly_from_points(final_points, binary.shape, band_mode)
    if safe_final_poly is not None:
        final_poly = safe_final_poly
    left_secondary_box, right_secondary_box = build_secondary_roi_boxes(binary.shape, final_points, final_poly)
    left_secondary_roi = limit_secondary_mask_by_global_poly(binary, left_secondary_box, final_poly)
    right_secondary_roi = limit_secondary_mask_by_global_poly(binary, right_secondary_box, final_poly)
    _, left_secondary_contour, left_secondary_ctr, _ = build_secondary_component_from_roi(binary, left_secondary_box, final_poly, roi_binary_override=left_secondary_roi)
    _, right_secondary_contour, right_secondary_ctr, _ = build_secondary_component_from_roi(binary, right_secondary_box, final_poly, roi_binary_override=right_secondary_roi)
    merged_pleura_mask, merged_pleura_contour, merged_pleura_ctr = merge_pleural_components(binary.shape, filtered_principal_ctr, left_secondary_ctr, right_secondary_ctr)
    _, limited_pleura_contour, limited_pleura_ctr = limit_pleura_thickness_by_poly(merged_pleura_mask, final_poly)
    ensure_dir(OUTPUT_DIR)
    save_important_identification_steps(index=index, crop=crop, palette=palette, binary=binary, traveler_points=traveler_points, selected_points=selected_points, band_points=band_points, band_info=band_info, filtered_points=filtered_points, poly_line=poly_line, roi_box=roi_box, principal_component_contour=principal_component_contour, filtered_principal_contour=filtered_principal_contour, final_poly=final_poly, left_secondary_contour=left_secondary_contour, right_secondary_contour=right_secondary_contour, merged_pleura_contour=merged_pleura_contour, limited_pleura_contour=limited_pleura_contour)
    save_final_contour_on_original(index, image_bgr, crop_box, limited_pleura_contour)
    return True

def main_batch():
    if RESET_OUTPUT_DIR_ON_RUN:
        reset_output_dir(OUTPUT_DIR)
    else:
        ensure_dir(OUTPUT_DIR)
    total_images = END_IDX - START_IDX
    for progress_index, index in enumerate(range(START_IDX, END_IDX), start=1):
        print(f'[{progress_index}/{total_images}] Imagine {index}')
        save_crop(index)

def main_single():
    if RESET_OUTPUT_DIR_ON_RUN:
        reset_output_dir(OUTPUT_DIR)
    else:
        ensure_dir(OUTPUT_DIR)
    print(f'[1/1] Imagine {SINGLE_IMAGE_IDX}')
    save_crop(SINGLE_IMAGE_IDX)
if __name__ == '__main__':
    if RUN_SINGLE_IMAGE:
        main_single()
    else:
        main_batch()
