import os
import math
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


INPUT_DIR = r"C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES"
OUTPUT_ROOT = r"C:\Facultate\AN4\Licenta\Licenta-Cod\DEBUG_OUT_HOUGH_ONLY"
FINAL_CONTOURS_DIR = os.path.join(OUTPUT_ROOT, "FINAL_CONTOURS_ONLY")
FINAL_MASKS_DIR = os.path.join(OUTPUT_ROOT, "FINAL_MASKS_ONLY")

START_IDX = 0
END_IDX = 61
SINGLE_IMAGE_IDX = 1
RUN_SINGLE_IMAGE = True
SHOW_RESULT = True
SAVE_DEBUG = True

PALETTE_COLORS = 7

MAX_PIXEL_VALUE = 255
MIN_CROP_SIZE = 30
DEFAULT_LEFT_BOUND = 25
CROP_RIGHT_MARGIN = 20
DEFAULT_BINARY_THRESHOLD = 128
BINARY_THRESHOLD = 110
MIN_BORDER_AREA = 1000
SMALL_CONTOUR_AREA = 30
LEFT_MARGIN_SEARCH_LIMIT = 100
HORIZONTAL_KERNEL_SIZE = (2, 1)
MORPH_ITERATIONS = 3


@dataclass
class PleuraLinePrior:
    x1: int
    y1: int
    x2: int
    y2: int
    slope: float
    intercept: float
    angle: float
    length: float
    support: float
    mean_intensity: float
    score: float

    def y_at(self, x_values):
        return self.slope * x_values + self.intercept


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_rgb(path, img_rgb):
    if img_rgb.ndim == 2:
        cv2.imwrite(path, img_rgb)
    else:
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def load_image_rgb(path):
    img = io.imread(path)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img.astype(np.uint8)


def to_gray_uint8(image):
    if image.ndim == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def crop_border(orig_image):
    gray_image = to_gray_uint8(orig_image)

    _, img_bin = cv2.threshold(
        gray_image,
        DEFAULT_BINARY_THRESHOLD,
        MAX_PIXEL_VALUE,
        cv2.THRESH_BINARY
    )

    _, threshold = cv2.threshold(
        gray_image,
        BINARY_THRESHOLD,
        MAX_PIXEL_VALUE,
        cv2.THRESH_BINARY
    )

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

            if len(approx) in [2, 4]:
                cv2.drawContours(black, [approx], 0, 255, -1)

    white_pixels = np.array(np.where(black == 255))

    if white_pixels.shape[1] == 0:
        left_bound = DEFAULT_LEFT_BOUND
    else:
        last_small = white_pixels[1, white_pixels[1] < LEFT_MARGIN_SEARCH_LIMIT]

        if len(last_small) == 0:
            left_bound = DEFAULT_LEFT_BOUND
        else:
            left_bound = int(last_small[-1])

    black = np.zeros_like(img_bin)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < SMALL_CONTOUR_AREA:
            approx = cv2.approxPolyDP(
                cnt,
                0.00001 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) in [2, 4]:
                cv2.drawContours(black, [approx], 0, 255, -1)

    hori_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        HORIZONTAL_KERNEL_SIZE
    )

    horizontal_lines_img = cv2.erode(
        black,
        hori_kernel,
        iterations=MORPH_ITERATIONS
    )

    horizontal_lines_img = cv2.dilate(
        horizontal_lines_img,
        hori_kernel,
        iterations=MORPH_ITERATIONS
    )

    columns = np.count_nonzero(horizontal_lines_img, axis=0)

    if len(columns) == 0 or np.max(columns) == 0:
        return gray_image.copy(), (0, 0, gray_image.shape[0], gray_image.shape[1])

    bar_pos = int(np.argmax(columns))
    bar = horizontal_lines_img[:, bar_pos] // 255
    bar_pixels = np.where(bar == 1)[0]

    if len(bar_pixels) == 0:
        return gray_image.copy(), (0, 0, gray_image.shape[0], gray_image.shape[1])

    top = int(bar_pixels[0])
    bottom = int(bar_pixels[-1])
    right = int(bar_pos - CROP_RIGHT_MARGIN)
    left = int(left_bound)

    if bottom <= top or right <= left:
        return gray_image.copy(), (0, 0, gray_image.shape[0], gray_image.shape[1])

    crop_img = gray_image[top:bottom, left:right].copy()

    if crop_img.size == 0 or crop_img.shape[0] < MIN_CROP_SIZE or crop_img.shape[1] < MIN_CROP_SIZE:
        return gray_image.copy(), (0, 0, gray_image.shape[0], gray_image.shape[1])

    return crop_img, (top, left, bottom, right)


def reduce_color_palette(image, nr_of_colors=PALETTE_COLORS):
    gray = to_gray_uint8(image)
    flat = gray.reshape(-1)

    if len(flat) == 0:
        return gray

    order = np.argsort(flat)
    sorted_pixels = flat[order]
    out = flat.copy()
    pixels_per_color = max(1, len(sorted_pixels) // nr_of_colors)

    for group_idx in range(nr_of_colors):
        start = group_idx * pixels_per_color
        end = (group_idx + 1) * pixels_per_color

        if group_idx == nr_of_colors - 1:
            end = len(sorted_pixels)

        if start >= len(sorted_pixels):
            continue

        group_indices = order[start:end]
        representative = int(sorted_pixels[min(end - 1, len(sorted_pixels) - 1)])
        out[group_indices] = representative

    return out.reshape(gray.shape).astype(np.uint8)


def build_total_white_mask(gray_image):
    gray = to_gray_uint8(gray_image)
    palette = reduce_color_palette(gray, PALETTE_COLORS)

    _, mask_otsu = cv2.threshold(
        palette,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    foreground_ratio = np.count_nonzero(mask_otsu) / max(mask_otsu.size, 1)

    if foreground_ratio < 0.003 or foreground_ratio > 0.45:
        threshold = np.percentile(palette, 88)
        mask = np.zeros_like(palette, dtype=np.uint8)
        mask[palette >= threshold] = 255
    else:
        mask = mask_otsu

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return mask, palette


def line_support(mask, x1, y1, x2, y2, thickness):
    h, w = mask.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.line(
        line_mask,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        255,
        thickness=max(1, int(thickness))
    )

    line_pixels = np.count_nonzero(line_mask)

    if line_pixels == 0:
        return 0.0

    hit_pixels = np.count_nonzero((line_mask > 0) & (mask > 0))
    return hit_pixels / line_pixels


def mean_intensity_on_line(gray, x1, y1, x2, y2, thickness):
    h, w = gray.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.line(
        line_mask,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        255,
        thickness=max(1, int(thickness))
    )

    vals = gray[line_mask > 0]

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals)) / 255.0



def build_bottom_edge_mask(mask):
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    edge = np.zeros((h, w), dtype=np.uint8)

    try:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    except Exception:
        return edge

    min_area = max(12, int(0.000015 * h * w))

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        ys, xs = np.where(labels == label)

        if len(xs) < 3:
            continue

        for x in np.unique(xs):
            y_vals = ys[xs == x]

            if len(y_vals) == 0:
                continue

            y_bottom = int(np.max(y_vals))
            edge[y_bottom, int(x)] = 255

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, int(0.012 * w)), 1)
    )
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    edge = cv2.dilate(edge, kernel_dilate, iterations=1)

    return edge


def sample_line_band(mask, gray, slope, intercept, x1, x2, offsets):
    h, w = mask.shape[:2]
    x_start = max(0, min(int(x1), int(x2)))
    x_end = min(w - 1, max(int(x1), int(x2)))

    if x_end <= x_start:
        return 0.0, 0.0

    xs = np.arange(x_start, x_end + 1, dtype=np.int32)
    values_mask = []
    values_gray = []

    for off in offsets:
        ys = np.rint(slope * xs + intercept + off).astype(np.int32)
        valid = (ys >= 0) & (ys < h)

        if not np.any(valid):
            continue

        values_mask.append(mask[ys[valid], xs[valid]] > 0)
        values_gray.append(gray[ys[valid], xs[valid]].astype(np.float32) / 255.0)

    if len(values_mask) == 0:
        return 0.0, 0.0

    mask_vals = np.concatenate(values_mask)
    gray_vals = np.concatenate(values_gray)

    return float(np.mean(mask_vals)), float(np.mean(gray_vals))


def smooth_1d(values, window):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    window = int(window)

    if window < 3 or len(values) < window:
        return values.copy()

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def projection_profile_around_line(mask, gray_image, slope, intercept, x1, x2, max_offset):
    h, w = mask.shape[:2]
    gray = to_gray_uint8(gray_image)

    x_start = max(0, min(int(x1), int(x2)))
    x_end = min(w - 1, max(int(x1), int(x2)))

    if x_end <= x_start:
        return None

    xs = np.arange(x_start, x_end + 1, dtype=np.int32)
    offsets = np.arange(-max_offset, max_offset + 1, dtype=np.int32)

    gray_profile = []
    mask_profile = []

    for off in offsets:
        ys = np.rint(slope * xs + intercept + int(off)).astype(np.int32)
        valid = (ys >= 0) & (ys < h)

        if not np.any(valid):
            gray_profile.append(0.0)
            mask_profile.append(0.0)
            continue

        gray_vals = gray[ys[valid], xs[valid]].astype(np.float32) / 255.0
        mask_vals = (mask[ys[valid], xs[valid]] > 0).astype(np.float32)

        gray_profile.append(float(np.mean(gray_vals)))
        mask_profile.append(float(np.mean(mask_vals)))

    gray_profile = np.asarray(gray_profile, dtype=np.float32)
    mask_profile = np.asarray(mask_profile, dtype=np.float32)
    smooth_window = max(3, int(0.008 * w))
    smooth_gray = smooth_1d(gray_profile, smooth_window)

    return {
        "offsets": offsets,
        "gray_profile": gray_profile,
        "smooth_gray": smooth_gray,
        "mask_profile": mask_profile
    }


def find_projection_peak_and_troughs(profile_data):
    if profile_data is None:
        return None

    offsets = profile_data["offsets"]
    smooth_gray = profile_data["smooth_gray"]

    if len(offsets) < 5:
        return None

    peak_idx = int(np.argmax(smooth_gray))
    peak_offset = int(offsets[peak_idx])
    peak_value = float(smooth_gray[peak_idx])

    if peak_idx <= 1 or peak_idx >= len(offsets) - 2:
        return None

    left_region = smooth_gray[:peak_idx]
    right_region = smooth_gray[peak_idx + 1:]

    if len(left_region) == 0 or len(right_region) == 0:
        return None

    left_idx = int(np.argmin(left_region))
    right_idx = int(peak_idx + 1 + np.argmin(right_region))

    left_offset = int(offsets[left_idx])
    right_offset = int(offsets[right_idx])

    if left_offset > right_offset:
        left_offset, right_offset = right_offset, left_offset

    left_min = float(smooth_gray[left_idx])
    right_min = float(smooth_gray[right_idx])
    background = 0.5 * (left_min + right_min)
    prominence = peak_value - background
    band_width = int(right_offset - left_offset + 1)

    return {
        "peak_idx": peak_idx,
        "peak_offset": peak_offset,
        "peak_value": peak_value,
        "left_offset": left_offset,
        "right_offset": right_offset,
        "left_min": left_min,
        "right_min": right_min,
        "background": background,
        "prominence": float(prominence),
        "band_width": band_width
    }


def draw_projection_debug(profile_data, peak_data, out_path):
    if profile_data is None or peak_data is None:
        return

    try:
        plt.figure(figsize=(8, 4))
        offsets = profile_data["offsets"]
        plt.plot(offsets, profile_data["gray_profile"], label="gray profile", alpha=0.45)
        plt.plot(offsets, profile_data["smooth_gray"], label="smoothed profile")
        plt.axvline(peak_data["peak_offset"], color="green", linestyle="--", label="peak")
        plt.axvline(peak_data["left_offset"], color="red", linestyle=":", label="left trough")
        plt.axvline(peak_data["right_offset"], color="red", linestyle=":", label="right trough")
        plt.xlabel("offset fata de linia candidata")
        plt.ylabel("intensitate medie")
        plt.title("Profil de proiectie ca in articol: peak + doua minime")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
    except Exception:
        pass


def evaluate_hough_line_candidate(mask, gray_image, x1, y1, x2, y2, source_name="hough"):
    h, w = mask.shape[:2]
    gray = to_gray_uint8(gray_image)

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    if abs(dx) < 1:
        return None

    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
        dx = float(x2 - x1)
        dy = float(y2 - y1)

    length = math.sqrt(dx * dx + dy * dy)
    length_frac = length / max(w, 1)

    if length_frac < 0.035:
        return None

    angle = float(math.degrees(math.atan2(dy, dx)))

    if abs(angle) > 35:
        return None

    slope = dy / dx
    intercept = y1 - slope * x1
    y_mid = (y1 + y2) / 2.0
    y_frac = y_mid / max(h, 1)

    if y_frac < 0.05 or y_frac > 0.94:
        return None

    # Articolul nu alege pur si simplu linia Hough/Radon.
    # Dupa gasirea liniei, o foloseste ca directie, apoi face o proiectie
    # laterala/verticala a intensitatii ca sa gaseasca peak-ul pleural si
    # cele doua minime din jurul lui. Asta facem aici, fara bottom-edge.
    max_offset = max(28, int(0.095 * h))
    profile_data = projection_profile_around_line(mask, gray, slope, intercept, x1, x2, max_offset)
    peak_data = find_projection_peak_and_troughs(profile_data)

    if peak_data is None:
        return None

    peak_offset = int(peak_data["peak_offset"])
    refined_intercept = intercept + peak_offset
    refined_y1 = int(np.clip(slope * x1 + refined_intercept, 0, h - 1))
    refined_y2 = int(np.clip(slope * x2 + refined_intercept, 0, h - 1))
    refined_y_mid = 0.5 * (refined_y1 + refined_y2)
    refined_y_frac = refined_y_mid / max(h, 1)

    line_thickness = max(3, int(0.008 * h))
    support = line_support(mask, x1, refined_y1, x2, refined_y2, thickness=line_thickness)
    intensity = mean_intensity_on_line(gray, x1, refined_y1, x2, refined_y2, thickness=line_thickness)

    peak_value = float(peak_data["peak_value"])
    prominence = float(peak_data["prominence"])
    band_width = float(peak_data["band_width"])

    # Densitatea in banda dintre cele doua minime. Daca banda e uriasa si plina,
    # probabil candidatul trece printr-o masa alba/fascie, nu printr-o creasta fina.
    trough_offsets = np.arange(peak_data["left_offset"], peak_data["right_offset"] + 1, dtype=np.int32)
    band_density, band_gray = sample_line_band(mask, gray, slope, refined_intercept, x1, x2, trough_offsets)

    # Verificam si ce se intampla imediat sub peak. Pleura trebuie sa fie o creasta
    # luminoasa; daca sub ea ramane aceeasi masa alba compacta, scadem scorul.
    below_start = peak_data["right_offset"] + max(3, int(0.008 * h))
    below_stop = below_start + max(8, int(0.025 * h))
    below_density, below_gray = sample_line_band(mask, gray, slope, intercept, x1, x2, np.arange(below_start, below_stop + 1))

    score = 0.0
    score += 4.2 * min(1.0, length_frac / 0.45)
    score += 3.5 * support
    score += 4.5 * peak_value
    score += 8.0 * max(0.0, prominence)
    score += 1.5 * band_gray
    score -= 1.5 * abs(angle) / 35.0
    score -= 1.2 * max(0.0, band_width / max(18, int(0.035 * h)) - 1.0)
    score -= 1.3 * max(0.0, band_density - 0.62)
    score -= 1.0 * max(0.0, below_density - 0.50)

    if refined_y_frac < 0.10 or refined_y_frac > 0.86:
        score -= 1.0

    prior = PleuraLinePrior(
        x1=int(x1),
        y1=int(refined_y1),
        x2=int(x2),
        y2=int(refined_y2),
        slope=float(slope),
        intercept=float(refined_intercept),
        angle=float(angle),
        length=float(length),
        support=float(support),
        mean_intensity=float(peak_value),
        score=float(score)
    )

    prior.source = source_name
    prior.length_frac = float(length_frac)
    prior.total_support = float(support)
    prior.core_density = float(support)
    prior.above_density = 0.0
    prior.below_density = float(below_density)
    prior.transition = float(prominence)
    prior.gray_transition = float(prominence)
    prior.wide_density = float(band_density)
    prior.y_frac = float(refined_y_frac)
    prior.above_gray = float(peak_data["left_min"])
    prior.below_gray = float(below_gray)
    prior.ridge_contrast = float(prominence)
    prior.projection_profile = profile_data
    prior.projection_peak = peak_data
    prior.peak_offset = int(peak_offset)
    prior.trough_left = int(peak_data["left_offset"] - peak_offset)
    prior.trough_right = int(peak_data["right_offset"] - peak_offset)
    prior.absolute_trough_left = int(peak_data["left_offset"])
    prior.absolute_trough_right = int(peak_data["right_offset"])
    prior.band_width = int(band_width)

    return prior

def deduplicate_line_candidates(candidates, image_shape):
    if len(candidates) == 0:
        return []

    h, w = image_shape[:2]
    kept = []

    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = False

        for old in kept:
            x_mid = int(0.5 * (max(cand.x1, old.x1) + min(cand.x2, old.x2)))
            x_mid = max(0, min(w - 1, x_mid))

            y_c = cand.slope * x_mid + cand.intercept
            y_o = old.slope * x_mid + old.intercept

            if abs(y_c - y_o) < max(8, int(0.025 * h)) and abs(cand.angle - old.angle) < 4:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

        if len(kept) >= 20:
            break

    return kept


def line_overlap_fraction(a, b):
    a_x1 = min(a.x1, a.x2)
    a_x2 = max(a.x1, a.x2)
    b_x1 = min(b.x1, b.x2)
    b_x2 = max(b.x1, b.x2)

    overlap = max(0, min(a_x2, b_x2) - max(a_x1, b_x1))
    min_len = max(1, min(a_x2 - a_x1, b_x2 - b_x1))

    return overlap / min_len, max(a_x1, b_x1), min(a_x2, b_x2)


def apply_fascia_aware_scoring(candidates, image_shape):
    """
    Ajusteaza scorul liniilor candidate ca sa evite fascia.

    Ideea este generala/anatomica:
    - fascia apare frecvent ca linie luminoasa aproape paralela deasupra;
    - pleura este mai probabil linia inferioara dintr-un grup de linii paralele;
    - daca o linie are alta linie similara sub ea, linia de sus este penalizata.
    """
    if len(candidates) == 0:
        return []

    h, w = image_shape[:2]
    min_vertical_gap = max(8, int(0.018 * h))
    max_vertical_gap = max(70, int(0.22 * h))
    min_overlap_frac = 0.22
    max_angle_diff = 8.0

    for cand in candidates:
        cand.raw_score = float(cand.score)
        cand.fascia_penalty = 0.0
        cand.lower_line_bonus = 0.0
        cand.parallel_below_count = 0
        cand.parallel_above_count = 0
        cand.strongest_parallel_below = 0.0

    for cand in candidates:
        for other in candidates:
            if other is cand:
                continue

            if abs(cand.angle - other.angle) > max_angle_diff:
                continue

            overlap_frac, ox1, ox2 = line_overlap_fraction(cand, other)

            if overlap_frac < min_overlap_frac:
                continue

            x_mid = 0.5 * (ox1 + ox2)
            y_cand = float(cand.y_at(x_mid))
            y_other = float(other.y_at(x_mid))
            vertical_gap = y_other - y_cand

            other_quality = (
                0.45 * float(getattr(other, "support", 0.0))
                + 0.35 * float(getattr(other, "ridge_contrast", 0.0))
                + 0.20 * float(getattr(other, "mean_intensity", 0.0))
            )

            comparable = (
                float(getattr(other, "raw_score", other.score)) >= float(getattr(cand, "raw_score", cand.score)) - 3.0
                or other_quality >= 0.22
            )

            if min_vertical_gap <= vertical_gap <= max_vertical_gap and comparable:
                cand.parallel_below_count += 1
                cand.strongest_parallel_below = max(cand.strongest_parallel_below, other_quality)

                # Daca exista linie similara sub candidat, candidatul de sus seamana cu fascia.
                cand.fascia_penalty += 3.0 + 2.0 * min(1.0, overlap_frac)

                # Linia de jos primeste un bonus mic, pentru ca poate fi pleura.
                other.parallel_above_count += 1
                other.lower_line_bonus += 1.2 + 0.9 * min(1.0, overlap_frac)

    for cand in candidates:
        y_frac = float(getattr(cand, "y_frac", ((cand.y1 + cand.y2) / 2.0) / max(h, 1)))

        # Nu respingem automat liniile de sus, dar le penalizam usor.
        if y_frac < 0.18:
            cand.fascia_penalty += 1.8
        elif y_frac < 0.26:
            cand.fascia_penalty += 0.75

        # Daca linia este foarte sus si are alta linie sub ea, penalizarea devine serioasa.
        if cand.parallel_below_count > 0 and y_frac < 0.36:
            cand.fascia_penalty += 2.0

        cand.score = float(cand.raw_score - cand.fascia_penalty + cand.lower_line_bonus)

    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    return candidates


def make_line_prior_from_points(points, mask, gray_image):
    if points is None or len(points) < 2:
        return None

    h, w = mask.shape[:2]
    pts = np.asarray(points, dtype=np.float64)
    xs = pts[:, 0]
    ys = pts[:, 1]

    if len(np.unique(xs)) < 2:
        return None

    try:
        coeff = np.polyfit(xs, ys, 1)
    except Exception:
        return None

    slope = float(coeff[0])
    intercept = float(coeff[1])
    angle = float(math.degrees(math.atan(slope)))

    if abs(angle) > 45:
        return None

    x1 = int(max(0, np.min(xs)))
    x2 = int(min(w - 1, np.max(xs)))

    if x2 <= x1:
        return None

    y1 = int(np.clip(slope * x1 + intercept, 0, h - 1))
    y2 = int(np.clip(slope * x2 + intercept, 0, h - 1))

    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    support = line_support(mask, x1, y1, x2, y2, thickness=max(3, int(0.015 * h)))
    intensity = mean_intensity_on_line(to_gray_uint8(gray_image), x1, y1, x2, y2, thickness=max(3, int(0.015 * h)))
    length_frac = length / max(w, 1)
    y_frac = ((y1 + y2) / 2.0) / max(h, 1)

    score = 0.0
    score += 5.0 * min(1.0, length_frac / 0.40)
    score += 4.0 * support
    score += 2.0 * intensity
    score -= 1.0 * abs(angle) / 45.0

    if 0.12 <= y_frac <= 0.78:
        score += 1.0

    return PleuraLinePrior(
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        slope=float(slope),
        intercept=float(intercept),
        angle=float(angle),
        length=float(length),
        support=float(support),
        mean_intensity=float(intensity),
        score=float(score)
    )


def estimate_pleura_line_from_components(mask, gray_image):
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    try:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    except Exception:
        return None

    candidates = []
    min_area = max(8, int(0.00001 * h * w))

    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        if bw < max(18, int(0.025 * w)):
            continue

        if bh > max(90, int(0.30 * h)):
            continue

        y_center_frac = (y + bh / 2.0) / max(h, 1)

        if y_center_frac < 0.08 or y_center_frac > 0.85:
            continue

        component_pixels = np.column_stack(np.where(labels == label))

        if len(component_pixels) < 5:
            continue

        pts_xy = np.column_stack((component_pixels[:, 1], component_pixels[:, 0]))
        prior = make_line_prior_from_points(pts_xy, mask, gray_image)

        if prior is None:
            continue

        aspect = bw / max(bh, 1)
        width_frac = bw / max(w, 1)
        height_frac = bh / max(h, 1)

        component_score = prior.score
        component_score += 2.5 * min(1.0, width_frac / 0.30)
        component_score += 1.5 * min(1.0, aspect / 8.0)
        component_score -= 1.5 * min(2.0, height_frac / 0.18)

        candidates.append((component_score, prior))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_prior = candidates[0]
    best_prior.score = float(best_score)
    return best_prior


def estimate_pleura_line_hough(mask, gray_image):
    h, w = mask.shape[:2]

    if h < 20 or w < 20:
        return None

    gray = to_gray_uint8(gray_image)

    candidate_sources = []

    # 1) Canny pe imaginea grayscale: cautam creste / linii luminoase interne,
    # nu marginea inferioara a obiectelor albe.
    canny_gray = cv2.Canny(gray, 25, 100)
    canny_gray[int(0.94 * h):, :] = 0
    candidate_sources.append(("canny_gray", canny_gray))

    # 2) Canny doar in zonele albe ale mastii, ca sa evitam zgomot din fundal.
    canny_inside_mask = cv2.bitwise_and(
        canny_gray,
        canny_gray,
        mask=(mask > 0).astype(np.uint8) * 255
    )
    candidate_sources.append(("canny_inside_mask", canny_inside_mask))

    # 3) Gradient orizontal/vertical din grayscale, util cand binarizarea face mase compacte.
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_y = np.uint8(np.clip(np.abs(grad_y), 0, 255))
    _, grad_mask = cv2.threshold(grad_y, int(np.percentile(grad_y, 88)), 255, cv2.THRESH_BINARY)
    grad_mask = cv2.bitwise_and(grad_mask, grad_mask, mask=(mask > 0).astype(np.uint8) * 255)
    grad_mask[int(0.94 * h):, :] = 0
    candidate_sources.append(("gradient_gray", grad_mask))

    # 4) Masca totala ramane sursa slaba, pentru cazul in care linia e foarte continua.
    candidate_sources.append(("total_mask", mask))

    param_sets = [
        (max(12, int(0.020 * w)), max(22, int(0.055 * w)), max(10, int(0.035 * w))),
        (max(9, int(0.015 * w)), max(16, int(0.040 * w)), max(16, int(0.055 * w))),
        (max(6, int(0.010 * w)), max(10, int(0.025 * w)), max(24, int(0.080 * w))),
    ]

    candidates = []

    for source_name, work_mask in candidate_sources:
        if work_mask is None or np.count_nonzero(work_mask) == 0:
            continue

        search_mask = work_mask.copy()

        # Stergem marginile extreme, dar fara sa presupunem unde este pleura.
        search_mask[:int(0.04 * h), :] = 0
        search_mask[int(0.96 * h):, :] = 0

        for threshold, min_line_length, max_line_gap in param_sets:
            lines = cv2.HoughLinesP(
                search_mask,
                rho=1,
                theta=np.pi / 180,
                threshold=threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )

            if lines is None:
                continue

            for item in lines:
                x1, y1, x2, y2 = item[0]
                cand = evaluate_hough_line_candidate(
                    mask,
                    gray,
                    x1,
                    y1,
                    x2,
                    y2,
                    source_name=source_name
                )

                if cand is not None:
                    candidates.append(cand)

    candidates = deduplicate_line_candidates(candidates, mask.shape)
    candidates = apply_fascia_aware_scoring(candidates, mask.shape)

    if len(candidates) == 0:
        fallback = estimate_pleura_line_from_components(mask, gray_image)

        if fallback is not None:
            fallback.source = "component_fallback"
            fallback.candidates = [fallback]
            return fallback

        return None

    best = candidates[0]
    best.candidates = candidates

    print("Top linii candidate Hough + proiectie ca in articol:")
    for idx, cand in enumerate(candidates[:10], start=1):
        print(
            "  #{} src={} score={:.3f} raw={:.3f} fascia_pen={:.3f} below={} above={} yfrac={:.3f} angle={:.2f} support={:.3f} peak={:.3f} prom={:.3f} band={:.3f} x=[{},{}] y=[{},{}]".format(
                idx,
                getattr(cand, "source", "?"),
                cand.score,
                getattr(cand, "raw_score", cand.score),
                getattr(cand, "fascia_penalty", 0.0),
                getattr(cand, "parallel_below_count", 0),
                getattr(cand, "parallel_above_count", 0),
                getattr(cand, "y_frac", 0.0),
                cand.angle,
                cand.support,
                cand.mean_intensity,
                getattr(cand, "ridge_contrast", 0.0),
                getattr(cand, "wide_density", 0.0),
                cand.x1,
                cand.x2,
                cand.y1,
                cand.y2
            )
        )

    return best


def moving_median_int(values, window):
    values = list(values)

    if len(values) == 0:
        return []

    window = int(window)

    if window < 3:
        return [int(round(v)) for v in values]

    if window % 2 == 0:
        window += 1

    radius = window // 2
    out = []

    for i in range(len(values)):
        a = max(0, i - radius)
        b = min(len(values), i + radius + 1)
        out.append(int(round(float(np.median(values[a:b])))))

    return out


def local_vertical_density(mask_bin, x, rows, radius_x=2, radius_y=1):
    h, w = mask_bin.shape[:2]
    vals = []

    x1 = max(0, int(x) - radius_x)
    x2 = min(w - 1, int(x) + radius_x)

    for r in rows:
        y1 = max(0, int(r) - radius_y)
        y2 = min(h - 1, int(r) + radius_y)
        patch = mask_bin[y1:y2 + 1, x1:x2 + 1]

        if patch.size == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.mean(patch > 0)))

    return np.asarray(vals, dtype=np.float32)


def track_curved_pleura_path(total_mask, gray_image, line_prior):
    """
    Hough/proiectia dau doar zona aproximativa. Curba finala se urmareste local,
    dar cu o regula importanta anti-fascie:

    - daca un punct are sub el un alt ridge luminos plauzibil, punctul de sus
      primeste penalizare;
    - se favorizeaza punctele care arata ca limita dintre tesutul de sus si
      zona mai aerata de jos;
    - nu se alege automat cel mai luminos pixel, pentru ca fascia este deseori
      mai luminoasa decat pleura.
    """
    h, w = total_mask.shape[:2]
    gray = to_gray_uint8(gray_image).astype(np.float32) / 255.0
    mask_bin = (total_mask > 0).astype(np.uint8)

    x_start = max(0, min(line_prior.x1, line_prior.x2))
    x_end = min(w - 1, max(line_prior.x1, line_prior.x2))

    if x_end <= x_start:
        return [], None

    xs = np.arange(x_start, x_end + 1, dtype=np.int32)

    profile_band = int(getattr(line_prior, "band_width", 0))
    search_radius = max(26, int(0.080 * h), profile_band * 2)
    search_radius = min(search_radius, max(32, int(0.20 * h)))

    jump_limit = max(6, int(0.016 * h))
    smooth_penalty = 0.075
    jump_penalty = 0.012

    min_below_gap = max(8, int(0.014 * h))
    max_below_gap = max(45, int(0.13 * h))

    column_rows = []
    column_scores = []

    def band_mean_col(arr, x, rows, off1, off2):
        vals = []
        x1 = max(0, int(x) - 2)
        x2 = min(w - 1, int(x) + 2)

        for r in rows:
            a = max(0, int(r) + off1)
            b = min(h - 1, int(r) + off2)

            if b < a:
                vals.append(0.0)
                continue

            patch = arr[a:b + 1, x1:x2 + 1]

            if patch.size == 0:
                vals.append(0.0)
            else:
                vals.append(float(np.mean(patch)))

        return np.asarray(vals, dtype=np.float32)

    for x in xs:
        y_line = int(round(line_prior.y_at(float(x))))
        y1 = max(0, y_line - search_radius)
        y2 = min(h - 1, y_line + search_radius)

        if y2 <= y1:
            column_rows.append(np.array([], dtype=np.int32))
            column_scores.append(np.array([], dtype=np.float32))
            continue

        rows = np.arange(y1, y2 + 1, dtype=np.int32)
        gray_vals = gray[rows, x]
        mask_vals = mask_bin[rows, x].astype(np.float32)
        density_vals = local_vertical_density(mask_bin, x, rows, radius_x=2, radius_y=1)

        up = np.maximum(rows - 2, 0)
        down = np.minimum(rows + 2, h - 1)
        local_contrast = gray_vals - 0.5 * (gray[up, x] + gray[down, x])
        local_contrast = np.maximum(local_contrast, 0.0)

        dist_to_hough = np.abs(rows.astype(np.float32) - float(y_line)) / max(search_radius, 1)

        above_mask = band_mean_col(mask_bin.astype(np.float32), x, rows, -max(18, int(0.035 * h)), -max(4, int(0.008 * h)))
        below_mask_near = band_mean_col(mask_bin.astype(np.float32), x, rows, max(4, int(0.008 * h)), max(20, int(0.040 * h)))
        below_gray_near = band_mean_col(gray, x, rows, max(5, int(0.010 * h)), max(28, int(0.055 * h)))

        # Pleura este mai probabil o limita: deasupra exista tesut/semnal,
        # dedesubt semnalul devine mai slab sau mai fragmentat.
        boundary_bonus = np.clip(above_mask - below_mask_near, -0.5, 1.0)
        below_compact_penalty = np.maximum(0.0, below_mask_near - 0.55)
        below_bright_penalty = np.maximum(0.0, below_gray_near - gray_vals - 0.03)

        base_strength = (
            0.85 * gray_vals
            + 0.65 * mask_vals
            + 0.25 * density_vals
            + 0.45 * local_contrast
        )

        # Penalizare anti-fascie: daca sub acest punct exista alt ridge plauzibil
        # in aceeasi coloana, punctul de sus nu trebuie sa castige doar pentru ca
        # este luminos.
        max_below_strength = np.zeros_like(base_strength, dtype=np.float32)

        for i, r in enumerate(rows):
            a = int(r) + min_below_gap
            b = int(r) + max_below_gap
            valid = (rows >= a) & (rows <= b)

            if np.any(valid):
                max_below_strength[i] = float(np.max(base_strength[valid]))

        ridge_below_penalty = np.maximum(0.0, max_below_strength - base_strength + 0.04)

        # Bonus mic pentru linia mai de jos cand exista alternative similare.
        lower_position_bonus = (rows.astype(np.float32) - float(y1)) / max(float(y2 - y1), 1.0)

        scores = (
            base_strength
            + 0.95 * boundary_bonus
            + 0.18 * lower_position_bonus
            - 0.22 * dist_to_hough
            - 1.10 * ridge_below_penalty
            - 0.85 * below_compact_penalty
            - 0.55 * below_bright_penalty
        )

        # Pixelii complet negri din masca pot ramane doar ca fallback, nu ca alegere principala.
        scores -= 0.55 * (1.0 - mask_vals)

        column_rows.append(rows)
        column_scores.append(scores.astype(np.float32))

    valid_cols = [i for i, rows in enumerate(column_rows) if len(rows) > 0]

    if len(valid_cols) < 2:
        return [], None

    dp = [None] * len(xs)
    parent = [None] * len(xs)

    first = valid_cols[0]
    dp[first] = column_scores[first].copy()
    parent[first] = np.full(len(column_rows[first]), -1, dtype=np.int32)

    prev_i = first

    for i in valid_cols[1:]:
        rows_cur = column_rows[i]
        scores_cur = column_scores[i]
        rows_prev = column_rows[prev_i]
        dp_prev = dp[prev_i]

        cur_dp = np.full(len(rows_cur), -1e9, dtype=np.float32)
        cur_parent = np.full(len(rows_cur), -1, dtype=np.int32)

        for ci, r in enumerate(rows_cur):
            dy = np.abs(rows_prev.astype(np.int32) - int(r))
            allowed = dy <= jump_limit

            if not np.any(allowed):
                best_pi = int(np.argmin(dy))
                transition = smooth_penalty * float(dy[best_pi]) + jump_penalty * float(dy[best_pi] ** 2) + 2.5
                val = float(dp_prev[best_pi]) - transition + float(scores_cur[ci])
            else:
                idxs = np.where(allowed)[0]
                transition = smooth_penalty * dy[idxs].astype(np.float32) + jump_penalty * (dy[idxs].astype(np.float32) ** 2)
                vals = dp_prev[idxs] - transition + float(scores_cur[ci])
                local_best = int(np.argmax(vals))
                best_pi = int(idxs[local_best])
                val = float(vals[local_best])

            cur_dp[ci] = val
            cur_parent[ci] = best_pi

        dp[i] = cur_dp
        parent[i] = cur_parent
        prev_i = i

    last = valid_cols[-1]
    best_idx = int(np.argmax(dp[last]))

    path = []
    i = last
    idx = best_idx

    while i >= first and idx >= 0:
        x = int(xs[i])
        y = int(column_rows[i][idx])
        path.append((x, y))

        pidx = int(parent[i][idx]) if parent[i] is not None else -1
        prev_valid = [v for v in valid_cols if v < i]

        if len(prev_valid) == 0:
            break

        i = prev_valid[-1]
        idx = pidx

    path.reverse()

    if len(path) < 2:
        return [], None

    px = [p[0] for p in path]
    py = [p[1] for p in path]
    smooth_window = max(5, int(0.016 * len(py)))
    py_smooth = moving_median_int(py, smooth_window)
    py_smooth = moving_median_int(py_smooth, max(3, smooth_window // 2))

    smooth_path = [(int(x), int(np.clip(y, 0, h - 1))) for x, y in zip(px, py_smooth)]

    debug = {
        "search_radius": int(search_radius),
        "jump_limit": int(jump_limit),
        "path_score": float(np.max(dp[last])),
        "anti_fascia": True
    }

    return smooth_path, debug

def build_pleura_mask_from_curved_path(total_mask, gray_image, path, line_prior):
    h, w = total_mask.shape[:2]
    gray = to_gray_uint8(gray_image).astype(np.float32) / 255.0
    mask_bin = (total_mask > 0).astype(np.uint8)
    pleura_mask = np.zeros((h, w), dtype=np.uint8)

    if path is None or len(path) < 2:
        return pleura_mask

    # Deocamdata pastram o banda moderata in jurul curbei, dar doar din pixeli reali
    # albi. Conturul final ramane bazat pe imagine, nu desenat artificial.
    keep_radius = max(4, int(0.010 * h))

    for x, y in path:
        y1 = max(0, int(y) - keep_radius)
        y2 = min(h - 1, int(y) + keep_radius)
        col_mask = mask_bin[y1:y2 + 1, int(x)] > 0

        if np.any(col_mask):
            rows = np.arange(y1, y2 + 1, dtype=np.int32)[col_mask]
            pleura_mask[rows, int(x)] = 255
        else:
            # Daca binarizarea a pierdut pixelii, alegem un mic punct luminos ca debug.
            rows = np.arange(y1, y2 + 1, dtype=np.int32)
            best = int(rows[int(np.argmax(gray[rows, int(x)]))])
            pleura_mask[best, int(x)] = 255

    # Conectare foarte usoara pe orizontala ca sa nu cada in pixeli rari.
    kernel_w = max(3, int(0.004 * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    pleura_mask = cv2.morphologyEx(pleura_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return pleura_mask


def build_pleura_mask_from_profile_band(total_mask, gray_image, line_prior):
    path, path_debug = track_curved_pleura_path(total_mask, gray_image, line_prior)
    line_prior.curved_path = path
    line_prior.curved_path_debug = path_debug

    if path is None or len(path) < 2:
        return np.zeros_like(total_mask, dtype=np.uint8)

    return build_pleura_mask_from_curved_path(total_mask, gray_image, path, line_prior)

def extract_components(mask):
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    min_area = max(6, int(0.000015 * h * w))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        comp_mask = np.zeros_like(mask, dtype=np.uint8)
        comp_mask[labels == label] = 255

        contours, _ = cv2.findContours(
            comp_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if contours is None or len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2)

        if len(pts) < 3:
            continue

        components.append({
            "label": label,
            "contour": contour,
            "mask": comp_mask,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "area": area,
            "cx": float(centroids[label][0]),
            "cy": float(centroids[label][1])
        })

    return components


def score_component_as_pleura(component, line_prior, image_shape, gray_image):
    h, w = image_shape[:2]
    contour = component["contour"]
    pts = contour.reshape(-1, 2)

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)

    expected_y = line_prior.y_at(xs)
    distances = np.abs(ys - expected_y)

    median_distance = float(np.median(distances))
    max_distance = float(np.percentile(distances, 90))

    width_frac = component["w"] / max(w, 1)
    height_frac = component["h"] / max(h, 1)
    area_frac = component["area"] / max(h * w, 1)
    aspect = component["w"] / max(component["h"], 1)

    if len(np.unique(xs)) > 1:
        try:
            comp_slope = float(np.polyfit(xs, ys, 1)[0])
        except Exception:
            comp_slope = 999.0
    else:
        comp_slope = 999.0

    line_band = max(10, int(0.055 * h))
    closeness = max(0.0, 1.0 - median_distance / max(line_band, 1))
    spread_penalty = max(0.0, max_distance / max(line_band, 1) - 1.0)
    slope_diff = abs(comp_slope - line_prior.slope)

    mean_intensity = 0.0
    vals = gray_image[component["mask"] > 0]

    if len(vals) > 0:
        mean_intensity = float(np.mean(vals)) / 255.0

    score = 0.0
    score += 5.0 * closeness
    score += 3.0 * min(1.0, width_frac / 0.22)
    score += 1.5 * min(1.0, aspect / 8.0)
    score += 1.5 * mean_intensity
    score -= 2.0 * min(2.0, height_frac / 0.14)
    score -= 1.2 * min(3.0, slope_diff)
    score -= 1.5 * spread_penalty

    return {
        "score": float(score),
        "median_distance": median_distance,
        "max_distance": max_distance,
        "width_frac": width_frac,
        "height_frac": height_frac,
        "area_frac": area_frac,
        "aspect": aspect,
        "slope": comp_slope,
        "mean_intensity": mean_intensity
    }


def select_pleura_components(mask, gray_image, line_prior):
    h, w = mask.shape[:2]
    components = extract_components(mask)
    selected = []
    rejected = []

    if line_prior is None:
        raise ValueError("Nu s-a gasit linie Hough pentru pleura.")

    if len(components) == 0:
        raise ValueError("Masca alba nu contine componente dupa connected components.")

    # Pragurile de mai jos nu mai sunt bariere tari. Ele sunt folosite doar
    # pentru prima trecere. Daca prima trecere nu gaseste nimic, alegem cei mai
    # buni candidati dupa scor, ca sa putem vedea debug-ul si sa reglam metoda.
    min_score = 2.2
    max_median_distance = max(18, int(0.110 * h))
    max_height_frac = 0.34
    min_width_frac = 0.003

    for component in components:
        metrics = score_component_as_pleura(component, line_prior, mask.shape, gray_image)
        component["metrics"] = metrics
        component["reject_reasons"] = []

        is_candidate = True

        if metrics["score"] < min_score:
            is_candidate = False
            component["reject_reasons"].append("low_score")

        if metrics["median_distance"] > max_median_distance:
            is_candidate = False
            component["reject_reasons"].append("far_from_hough_line")

        if metrics["height_frac"] > max_height_frac:
            is_candidate = False
            component["reject_reasons"].append("too_tall")

        if metrics["width_frac"] < min_width_frac:
            is_candidate = False
            component["reject_reasons"].append("too_narrow")

        if is_candidate:
            selected.append(component)
        else:
            rejected.append(component)

    # Fallback NOU, nu metoda veche: daca filtrarea stricta elimina tot,
    # pastram cei mai buni candidati dupa scor. Altfel codul se opreste si nu
    # putem vedea ce face metoda noua pe imaginea respectiva.
    if len(selected) == 0:
        ranked = sorted(
            components,
            key=lambda c: c["metrics"]["score"],
            reverse=True
        )

        best_score = ranked[0]["metrics"]["score"]
        relaxed_distance = max(35, int(0.180 * h))
        relaxed_height = 0.50

        selected = []
        rejected = []

        for component in ranked:
            metrics = component["metrics"]
            keep = True

            if metrics["score"] < best_score - 2.5:
                keep = False

            if metrics["median_distance"] > relaxed_distance:
                keep = False

            if metrics["height_frac"] > relaxed_height:
                keep = False

            if metrics["width_frac"] < 0.0015:
                keep = False

            if keep:
                component["selected_by"] = "relaxed_best_score"
                selected.append(component)
            else:
                rejected.append(component)

            if len(selected) >= 8:
                break

        if len(selected) == 0:
            # Ultimul fallback tot pe metoda noua: alegem doar componenta cu scor maxim,
            # ca sa salvam debug-ul si sa vedem unde a estimat Hough pleura.
            ranked[0]["selected_by"] = "top1_debug_fallback"
            selected = [ranked[0]]
            rejected = ranked[1:]

    selected = sorted(selected, key=lambda c: c["x"])

    print("Componente albe totale:", len(components))
    print("Componente pleura selectate:", len(selected))
    print("Top componente dupa scor:")

    ranked_for_print = sorted(
        components,
        key=lambda c: c["metrics"]["score"],
        reverse=True
    )[:8]

    for idx, comp in enumerate(ranked_for_print, start=1):
        m = comp["metrics"]
        print(
            "  #{} score={:.3f} dist_med={:.1f} width_frac={:.3f} height_frac={:.3f} aspect={:.2f} x=[{},{}] y=[{},{}]".format(
                idx,
                m["score"],
                m["median_distance"],
                m["width_frac"],
                m["height_frac"],
                m["aspect"],
                comp["x"],
                comp["x"] + comp["w"],
                comp["y"],
                comp["y"] + comp["h"]
            )
        )

    return selected, rejected, components


def build_mask_from_components(components, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for component in components:
        cv2.drawContours(mask, [component["contour"]], -1, 255, -1)

    return mask


def contours_from_mask(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours is None:
        return []

    return contours


def draw_hough_line(image_rgb, line_prior, color=(255, 255, 0), thickness=2):
    out = image_rgb.copy()

    if line_prior is None:
        return out

    cv2.line(
        out,
        (int(line_prior.x1), int(line_prior.y1)),
        (int(line_prior.x2), int(line_prior.y2)),
        color,
        thickness
    )

    return out


def draw_components(image_rgb, components, color, thickness=2, draw_id=False):
    out = image_rgb.copy()

    for idx, component in enumerate(components, start=1):
        cv2.drawContours(out, [component["contour"]], -1, color, thickness)

        if draw_id:
            cv2.putText(
                out,
                str(idx),
                (int(component["cx"]), int(component["cy"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

    return out


def detect_pleura_hough_only(crop_gray):
    crop_gray = to_gray_uint8(crop_gray)

    total_mask, palette = build_total_white_mask(crop_gray)
    foreground_ratio = np.count_nonzero(total_mask) / max(total_mask.size, 1)
    print("Foreground alb in masca totala: {:.4f}".format(foreground_ratio))

    line_prior = estimate_pleura_line_hough(total_mask, crop_gray)

    if line_prior is None:
        raise ValueError("Nu s-a gasit nicio linie candidata plauzibila pentru pleura.")

    print(
        "Linie aleasa: src={} x1={} y1={} x2={} y2={} angle={:.2f} support={:.3f} core={:.3f} ridge={:.3f} score={:.3f} raw={:.3f} fascia_pen={:.3f} below={} above={}".format(
            getattr(line_prior, "source", "?"),
            line_prior.x1,
            line_prior.y1,
            line_prior.x2,
            line_prior.y2,
            line_prior.angle,
            line_prior.support,
            line_prior.mean_intensity,
            getattr(line_prior, "ridge_contrast", 0.0),
            line_prior.score,
            getattr(line_prior, "raw_score", line_prior.score),
            getattr(line_prior, "fascia_penalty", 0.0),
            getattr(line_prior, "parallel_below_count", 0),
            getattr(line_prior, "parallel_above_count", 0)
        )
    )

    pleura_mask = build_pleura_mask_from_profile_band(
        total_mask,
        crop_gray,
        line_prior
    )

    final_contours = contours_from_mask(pleura_mask)

    if len(final_contours) == 0:
        raise ValueError("Linia Hough a fost gasita, dar masca pleurei extrasa din profilul de intensitate este goala.")

    all_components = extract_components(total_mask)
    selected_components = extract_components(pleura_mask)

    for component in selected_components:
        component["metrics"] = score_component_as_pleura(component, line_prior, total_mask.shape, crop_gray)

    return {
        "crop_gray": crop_gray,
        "palette": palette,
        "total_mask": total_mask,
        "profile_search_mask": total_mask,
        "line_prior": line_prior,
        "line_candidates": getattr(line_prior, "candidates", []),
        "all_components": all_components,
        "selected_components": selected_components,
        "rejected_components": [],
        "pleura_mask": pleura_mask,
        "final_contours": final_contours
    }

def prepare_crop_gray(original_image):
    crop_gray, crop_box = crop_border(original_image)
    return crop_gray, crop_box


def save_debug_outputs(idx, crop_gray, result, output_root):
    debug_dir = ensure_dir(os.path.join(output_root, "DEBUG_HOUGH_ONLY"))
    prefix = format(idx, "02d")

    crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)

    save_rgb(os.path.join(debug_dir, prefix + "_00_crop.png"), crop_rgb)
    save_rgb(os.path.join(debug_dir, prefix + "_01_palette.png"), result["palette"])
    save_rgb(os.path.join(debug_dir, prefix + "_02_total_white_mask.png"), result["total_mask"])
    save_rgb(os.path.join(debug_dir, prefix + "_03_profile_search_mask.png"), result.get("profile_search_mask", result["total_mask"]))

    hough_overlay = draw_hough_line(crop_rgb, result["line_prior"])
    save_rgb(os.path.join(debug_dir, prefix + "_04_hough_line_aleasa.png"), hough_overlay)

    candidate_overlay = crop_rgb.copy()
    for cand in result.get("line_candidates", [])[:10]:
        candidate_overlay = draw_hough_line(candidate_overlay, cand, color=(255, 0, 0), thickness=1)
    candidate_overlay = draw_hough_line(candidate_overlay, result["line_prior"], color=(255, 255, 0), thickness=2)
    save_rgb(os.path.join(debug_dir, prefix + "_05_top_linii_candidate.png"), candidate_overlay)

    draw_projection_debug(
        getattr(result["line_prior"], "projection_profile", None),
        getattr(result["line_prior"], "projection_peak", None),
        os.path.join(debug_dir, prefix + "_05b_projection_profile_peak_troughs.png")
    )

    all_overlay = draw_components(crop_rgb, result["all_components"], (120, 120, 255), 1, draw_id=True)
    save_rgb(os.path.join(debug_dir, prefix + "_06_all_components.png"), all_overlay)

    selected_overlay = draw_components(crop_rgb, result["selected_components"], (0, 255, 0), 2, draw_id=True)
    selected_overlay = draw_hough_line(selected_overlay, result["line_prior"], color=(255, 255, 0), thickness=2)
    save_rgb(os.path.join(debug_dir, prefix + "_05_selected_pleura_components.png"), selected_overlay)

    rejected_overlay = draw_components(crop_rgb, result["rejected_components"], (255, 0, 0), 1, draw_id=False)
    save_rgb(os.path.join(debug_dir, prefix + "_06_rejected_components.png"), rejected_overlay)

    save_rgb(os.path.join(debug_dir, prefix + "_07_pleura_mask.png"), result["pleura_mask"])

    return debug_dir


def plot_result(img_name, crop_gray, result):
    crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)

    hough_overlay = draw_hough_line(crop_rgb, result["line_prior"])
    all_overlay = draw_components(crop_rgb, result["all_components"], (120, 120, 255), 1, draw_id=True)
    selected_overlay = draw_components(crop_rgb, result["selected_components"], (0, 255, 0), 2, draw_id=True)
    selected_overlay = draw_hough_line(selected_overlay, result["line_prior"], color=(255, 255, 0), thickness=2)

    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.imshow(crop_gray, cmap="gray")
    plt.title("crop dupa crop border")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(result["palette"], cmap="gray")
    plt.title("paleta redusa la 7 nuante")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(result["total_mask"], cmap="gray")
    plt.title("masca totala: tot ce e alb")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(hough_overlay)
    plt.title("linia Hough probabila")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(all_overlay)
    plt.title("toate componentele separate")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(selected_overlay)
    plt.title("componente selectate ca pleura")
    plt.axis("off")

    plt.suptitle(img_name + " - Hough-only pleura detection")
    plt.tight_layout()
    plt.show()


def print_detection_debug(result):
    line = result["line_prior"]

    print("\n--- HOUGH ONLY DEBUG ---")
    print("Linie Hough:")
    print(" angle={:.2f}".format(line.angle))
    print(" length={:.1f}".format(line.length))
    print(" support={:.3f}".format(line.support))
    print(" mean_intensity={:.3f}".format(line.mean_intensity))
    print(" score={:.3f}".format(line.score))
    print(" source=" + str(getattr(line, "source", "?")))
    print(" projection_peak_offset=" + str(getattr(line, "peak_offset", None)))
    print(" projection_troughs=[{}, {}]".format(getattr(line, "trough_left", None), getattr(line, "trough_right", None)))
    print(" peak_intensity={:.3f}".format(getattr(line, "mean_intensity", 0.0)))
    print(" peak_prominence={:.3f}".format(getattr(line, "ridge_contrast", 0.0)))
    print(" band_density={:.3f}".format(getattr(line, "wide_density", 0.0)))
    print(" below_density={:.3f}".format(getattr(line, "below_density", 0.0)))
    print(" raw_score={:.3f}".format(getattr(line, "raw_score", line.score)))
    print(" fascia_penalty={:.3f}".format(getattr(line, "fascia_penalty", 0.0)))
    print(" parallel_below_count=" + str(getattr(line, "parallel_below_count", 0)))
    print(" parallel_above_count=" + str(getattr(line, "parallel_above_count", 0)))
    print("Componente totale albe: " + str(len(result["all_components"])))
    print("Componente contur pleura extrase din profil intensitate: " + str(len(result["selected_components"])))

    for idx, component in enumerate(result["selected_components"], start=1):
        m = component.get("metrics", {})
        print(
            "selected_" + str(idx)
            + " | score={:.3f}".format(m.get("score", 0.0))
            + " | dist={:.1f}".format(m.get("median_distance", 0.0))
            + " | width_frac={:.3f}".format(m.get("width_frac", 0.0))
            + " | height_frac={:.3f}".format(m.get("height_frac", 0.0))
            + " | aspect={:.2f}".format(m.get("aspect", 0.0))
            + " | intensity={:.3f}".format(m.get("mean_intensity", 0.0))
        )



def draw_final_contours_overlay(crop_gray, final_contours):
    crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)

    for contour in final_contours:
        try:
            cv2.drawContours(crop_rgb, [contour], -1, (0, 255, 0), 1)
        except Exception:
            pass

    return crop_rgb


def save_final_contour_outputs(idx, crop_gray, result, output_root):
    contours_dir = ensure_dir(FINAL_CONTOURS_DIR)
    masks_dir = ensure_dir(FINAL_MASKS_DIR)

    prefix = format(idx, "02d")

    final_overlay = draw_final_contours_overlay(
        crop_gray,
        result.get("final_contours", [])
    )

    save_rgb(
        os.path.join(contours_dir, prefix + "_final_contour.png"),
        final_overlay
    )

    save_rgb(
        os.path.join(masks_dir, prefix + "_final_mask.png"),
        result["pleura_mask"]
    )

    contour_only = np.zeros_like(result["pleura_mask"], dtype=np.uint8)
    for contour in result.get("final_contours", []):
        try:
            cv2.drawContours(contour_only, [contour], -1, 255, 1)
        except Exception:
            pass

    save_rgb(
        os.path.join(masks_dir, prefix + "_final_contour_pixel_only.png"),
        contour_only
    )

    return contours_dir


def run_one_image(idx, show_result=True, save_debug=True):
    img_name = str(idx) + ".jpg"
    img_path = os.path.join(INPUT_DIR, img_name)

    print("\nProcesez: " + img_name)

    if not os.path.exists(img_path):
        raise FileNotFoundError("Fisier inexistent: " + img_path)

    original_image = load_image_rgb(img_path)
    crop_gray, crop_box = prepare_crop_gray(original_image)

    print("Crop box: top={} left={} bottom={} right={} size={}".format(
        crop_box[0],
        crop_box[1],
        crop_box[2],
        crop_box[3],
        crop_gray.shape
    ))

    result = detect_pleura_hough_only(crop_gray)
    print_detection_debug(result)

    final_dir = save_final_contour_outputs(idx, crop_gray, result, OUTPUT_ROOT)
    print("Contur final salvat in: " + final_dir)

    if save_debug:
        debug_dir = save_debug_outputs(idx, crop_gray, result, OUTPUT_ROOT)
        print("Debug salvat in: " + debug_dir)

    if show_result:
        plot_result(img_name, crop_gray, result)

    return result


def main1():
    ensure_dir(OUTPUT_ROOT)

    total = END_IDX - START_IDX + 1
    success = 0
    failed = []

    for idx in range(START_IDX, END_IDX + 1):
        current = idx - START_IDX + 1
        print("\n[" + str(current) + "/" + str(total) + "]")

        try:
            run_one_image(idx, show_result=False, save_debug=SAVE_DEBUG)
            success += 1
        except Exception as e:
            failed.append((idx, str(e)))
            print("Eroare: " + str(e))

    print("\n--- REZUMAT ---")
    print("Reusite: " + str(success))
    print("Esuate: " + str(len(failed)))

    if len(failed) > 0:
        print("Imagini esuate:")
        for idx, err in failed:
            print(" - " + str(idx) + ".jpg: " + err)


def main2():
    run_one_image(
        SINGLE_IMAGE_IDX,
        show_result=SHOW_RESULT,
        save_debug=SAVE_DEBUG
    )


if __name__ == "__main__":
    if RUN_SINGLE_IMAGE:
        main2()
    else:
        main1()
