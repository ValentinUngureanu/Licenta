import math
import os
import re
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tes
from PIL import Image
from matplotlib.pyplot import figure
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage import img_as_ubyte, io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen

INPUT_DIR = r"C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES"
OUTPUT_ROOT = r"C:\Facultate\AN4\Licenta\Licenta-Cod\OUTPUT_INITIAL_BASELINE"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

START_IDX = 0
END_IDX = 61
SINGLE_IMAGE_IDX = 4
RUN_SINGLE_IMAGE = True

SAVE_BATCH_RESULTS = True
SHOW_SINGLE_RESULT = True
DEBUG_VERBOSE = True
SHOW_WIDTH_AND_INTERRUPTION_PLOTS = False
SAVE_DEBUG_MASKS = True
SHOW_DEBUG_MASKS = True
DEBUG_DRAW_COMPONENTS_SEPARATELY = False

ENABLE_NOISE_REMOVAL = True
NOISE_REMOVAL_SIZE = (100, 100)

# Praguri folosite pentru a reproduce fluxul din documentatie:
# - paleta redusa la 7 culori;
# - traveler pixels filtrati cu deviere maxima 10 fata de polinomul de grad 3;
# - componentele finale/secundare pastrate doar daca sunt la max 50 px fata de polinomul pleurei;
# - polinomul final al componentei principale este de grad 2.
DOC_PALETTE_COLORS = 7
DOC_TRAVELER_DEVIATION = 10
DOC_COMPONENT_DEVIATION = 50
DOC_FINAL_POLY_ORDER = 2

# Pasul de banda centrala a ramas in fisier doar ca fallback manual,
# dar fluxul activ foloseste IdentifyPrincipalContour(img) pe ROI-ul complet,
# conform documentatiei.
PRINCIPAL_SEARCH_BANDS = (
    (0.25, 0.85),
    (0.15, 0.95),
)

# CropBorder imbunatatit: aceste valori taie zonele in care apar de obicei
# elementele de UI ale ecografului: header, meniu lateral, scala si textul de jos.
# Sunt folosite doar ca margini minime; functia incearca in plus sa gaseasca automat
# bara de scala si limitele utile ale imaginii.
CROP_FORCE_TOP_FRAC = 0.06
CROP_FORCE_BOTTOM_FRAC = 0.03
CROP_FORCE_LEFT_FRAC = 0.05
CROP_FORCE_RIGHT_FRAC = 0.06
CROP_MIN_WIDTH_FRAC = 0.35
CROP_MIN_HEIGHT_FRAC = 0.25

LAST_CROP_TOP_OFFSET = 0
LAST_CROP_LEFT_OFFSET = 0
LAST_CROP_BOX = None


def NoiseRemoval(image, target_size=NOISE_REMOVAL_SIZE):
    image = np.asarray(image)

    if image.size == 0:
        return image.copy()

    h, w = image.shape[:2]

    if h <= 1 or w <= 1:
        return image.copy()

    target_w = int(target_size[0])
    target_h = int(target_size[1])

    if target_w <= 1 or target_h <= 1:
        return image.copy()

    down = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        down,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    if restored.dtype != image.dtype:
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            restored = np.clip(restored, info.min, info.max).astype(image.dtype)
        else:
            restored = restored.astype(image.dtype)

    return restored


### Crop image border:

def _crop_to_gray_uint8(image):
    image = np.asarray(image)

    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Imagine invalida pentru CropBorder: " + str(image.shape))

    if gray.dtype != np.uint8:
        gray = gray.astype(np.float32)
        if gray.size > 0 and np.max(gray) <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def _clamp_crop_box(top, bottom, left, right, h, w):
    top = int(max(0, min(top, h - 1)))
    bottom = int(max(top + 1, min(bottom, h)))
    left = int(max(0, min(left, w - 1)))
    right = int(max(left + 1, min(right, w)))
    return top, bottom, left, right


def _crop_box_is_valid(top, bottom, left, right, h, w):
    crop_h = bottom - top
    crop_w = right - left

    if crop_h < CROP_MIN_HEIGHT_FRAC * h:
        return False

    if crop_w < CROP_MIN_WIDTH_FRAC * w:
        return False

    return True


def _estimate_box_from_scale_and_borders(gray):
    h, w = gray.shape[:2]

    top = int(CROP_FORCE_TOP_FRAC * h)
    bottom = int((1.0 - CROP_FORCE_BOTTOM_FRAC) * h)
    left = int(CROP_FORCE_LEFT_FRAC * w)
    right = int((1.0 - CROP_FORCE_RIGHT_FRAC) * w)

    _, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_bin = np.zeros_like(gray)
    large_shape_mask = np.zeros_like(gray)
    small_shape_mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(
            cnt,
            0.01 * cv2.arcLength(cnt, True),
            True
        )

        if area > 1000 and (len(approx) == 4 or len(approx) == 2):
            cv2.drawContours(large_shape_mask, [approx], 0, 255, -1)

    white_pixels = np.array(np.where(large_shape_mask == 255))

    if white_pixels.shape[1] > 0:
        left_candidates = white_pixels[1, white_pixels[1] < int(0.20 * w)]

        if len(left_candidates) > 0:
            left = max(left, int(left_candidates[-1]) + 3)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 35:
            approx = cv2.approxPolyDP(
                cnt,
                0.00001 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4 or len(approx) == 2:
                cv2.drawContours(small_shape_mask, [approx], 0, 255, -1)

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    temp = cv2.erode(small_shape_mask, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(temp, hori_kernel, iterations=3)

    columns = np.count_nonzero(horizontal_lines_img, axis=0)

    if columns.size > 0 and np.max(columns) > 0:
        bar_pos = int(np.argmax(columns))

        if bar_pos > int(0.25 * w):
            right = min(right, bar_pos - 20)

        bar = horizontal_lines_img[:, bar_pos] // 255
        bar_pixels = np.where(bar == 1)[0]

        if len(bar_pixels) > 5:
            scale_top = int(bar_pixels[0])
            scale_bottom = int(bar_pixels[-1])

            if 0 < scale_top < h and 0 < scale_bottom < h and scale_bottom > scale_top:
                top = max(top, scale_top)
                bottom = min(bottom, scale_bottom)

    return _clamp_crop_box(top, bottom, left, right, h, w)


def _refine_box_by_ui_density(gray, top, bottom, left, right):
    h, w = gray.shape[:2]
    top, bottom, left, right = _clamp_crop_box(top, bottom, left, right, h, w)

    crop = gray[top:bottom, left:right]

    if crop.size == 0:
        return top, bottom, left, right

    crop_h, crop_w = crop.shape[:2]

    bright = crop > 160
    col_density = np.count_nonzero(bright, axis=0) / max(crop_h, 1)
    row_density = np.count_nonzero(bright, axis=1) / max(crop_w, 1)

    if crop_w >= 80:
        win = max(15, int(0.025 * crop_w))
        kernel = np.ones(win, dtype=np.float32) / win
        smooth_col = np.convolve(col_density, kernel, mode="same")

        left_scan_end = max(10, int(0.22 * crop_w))
        right_scan_start = min(crop_w - 10, int(0.78 * crop_w))

        left_candidates = np.where(smooth_col[:left_scan_end] < 0.010)[0]

        if len(left_candidates) > 0:
            new_left_local = int(left_candidates[0])
            left = max(left, left + new_left_local - 3)

        right_candidates = np.where(smooth_col[right_scan_start:] < 0.010)[0]

        if len(right_candidates) > 0:
            new_right_local = right_scan_start + int(right_candidates[-1])
            right = min(right, left + new_right_local + 3)

    if crop_h >= 80:
        win = max(9, int(0.025 * crop_h))
        kernel = np.ones(win, dtype=np.float32) / win
        smooth_row = np.convolve(row_density, kernel, mode="same")

        top_scan_end = max(10, int(0.18 * crop_h))
        bottom_scan_start = min(crop_h - 10, int(0.82 * crop_h))

        top_candidates = np.where(smooth_row[:top_scan_end] < 0.015)[0]

        if len(top_candidates) > 0:
            new_top_local = int(top_candidates[0])
            top = max(top, top + new_top_local - 2)

        bottom_candidates = np.where(smooth_row[bottom_scan_start:] < 0.015)[0]

        if len(bottom_candidates) > 0:
            new_bottom_local = bottom_scan_start + int(bottom_candidates[-1])
            bottom = min(bottom, top + new_bottom_local + 2)

    top, bottom, left, right = _clamp_crop_box(top, bottom, left, right, h, w)

    if not _crop_box_is_valid(top, bottom, left, right, h, w):
        return _estimate_box_from_scale_and_borders(gray)

    return top, bottom, left, right


def CropBorderWithOffset(orig_Image):
    global LAST_CROP_TOP_OFFSET
    global LAST_CROP_LEFT_OFFSET
    global LAST_CROP_BOX

    gray = _crop_to_gray_uint8(orig_Image)
    h, w = gray.shape[:2]

    top, bottom, left, right = _estimate_box_from_scale_and_borders(gray)
    top, bottom, left, right = _refine_box_by_ui_density(gray, top, bottom, left, right)

    if not _crop_box_is_valid(top, bottom, left, right, h, w):
        top = int(CROP_FORCE_TOP_FRAC * h)
        bottom = int((1.0 - CROP_FORCE_BOTTOM_FRAC) * h)
        left = int(CROP_FORCE_LEFT_FRAC * w)
        right = int((1.0 - CROP_FORCE_RIGHT_FRAC) * w)
        top, bottom, left, right = _clamp_crop_box(top, bottom, left, right, h, w)

    crop_img = gray[top:bottom, left:right].copy()

    LAST_CROP_TOP_OFFSET = int(top)
    LAST_CROP_LEFT_OFFSET = int(left)
    LAST_CROP_BOX = (int(top), int(bottom), int(left), int(right))

    if DEBUG_VERBOSE:
        print(
            "CropBorder box:",
            "top=", int(top),
            "bottom=", int(bottom),
            "left=", int(left),
            "right=", int(right),
            "size=", str(crop_img.shape)
        )

    return crop_img, int(top), int(left)


def CropBorder(orig_Image):
    crop_img, _, _ = CropBorderWithOffset(orig_Image)
    return crop_img


### Convert pixels to mm:

def PixelConverter(orig_Image):
    orig_img = orig_Image.copy()

    gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    (thresh, img_bin) = cv2.threshold(gray_Image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # figure(num=None, figsize=(20, 20))
    # plt.imshow(np.invert(img_bin), cmap = 'gray')

    # Converting image to a binary image
    # (black and white only image)
    _, threshold = cv2.threshold(gray_Image, 110, 255,
                                 cv2.THRESH_BINARY)

    # Detecting shapes in image by selecting region
    # with same colors or intensity
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Searching through every region selected to
    # find the required polygon
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area
        if area < 20:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)

            # Checking if the no. of sides of the selected region is 4
            if (len(approx) == 4):
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

        # Converting image to a binary image
        # (black and white only image)
        _, threshold = cv2.threshold(gray_Image, 110, 255,
                                     cv2.THRESH_BINARY)

        # Detecting shapes in image by selecting region
        # with same colors or intensity
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

    black = np.zeros_like(img_bin)

    # Searching through every region selected to
    # find the required polygon
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area
        if area < 20:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)

            # Checking if the no. of sides of the selected region is 4
            if (len(approx) == 4):
                cv2.drawContours(black, [approx], 0, 255, -1)

            # Checking if the no. of sides of the selected region is 2
            if (len(approx) == 2):
                cv2.drawContours(black, [approx], 0, 255, -1)

    img = black

    # Defining a kernel length
    kernel_length = 2

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(black, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    bar_image = horizontal_lines_img

    columns = np.zeros(bar_image.shape[1], dtype=int)

    for i in range(bar_image.shape[1]):
        columns[i] = np.count_nonzero(bar_image[:, i])

    bar_pos = np.where(columns == np.max(columns))
    bar_pos = bar_pos[0][0]

    # bar column as array
    bar = bar_image[:, bar_pos] // 255
    # bar = bar.tolist()

    indices = [i for i, x in enumerate(bar) if x == 1]  # indices of all nonzero pixels

    tes.pytesseract.tesseract_cmd = TESSERACT_CMD

    image_str = tes.image_to_string(np.invert(img_bin))  # all the text in the image
    # print(image_str)

    if 'cm' in image_str:
        depth_str = image_str[image_str.find('cm') - 5:image_str.find('cm')]
        depth_no = [int(i) for i in depth_str if i.isdigit()]
        depth = depth_no[0] * 10  # to transform it from cm to mm

    if 'mm' in image_str:
        depth_str = image_str[image_str.find('mm') - 5:image_str.find('mm')]
        depth_no = [int(s) for s in re.findall(r'\b\d+\b', depth_str)]
        depth = depth_no[0]

    if 'mm' not in image_str or 'cm' not in image_str:
        if '\nD ' in image_str:
            depth_str = image_str[image_str.find('\nD '):image_str.find('\nD ') + 6]
            depth_no = [int(i) for i in depth_str if i.isdigit()]
            if depth_no[0] < 50:
                depth = depth_no[0] * 10  # to transform it from cm to mm
            if depth_no[0] > 50:
                depth = depth_no[0]

        if '-D ' in image_str:
            depth_str = image_str[image_str.find('-D '):image_str.find('-D ') + 6]
            depth_no = [int(i) for i in depth_str if i.isdigit()]
            if depth_no[0] < 50:
                depth = depth_no[0] * 10  # to transform it from cm to mm
            if depth_no[0] > 50:
                depth = depth_no[0]

        if '\nD ' not in image_str and '-D ' not in image_str:
            print('Error: Ultrasound transducer depth not found! Please provide it manually')  # depth=?
            depth = np.sum(np.diff(indices))

    depth_pix = depth / np.sum(np.diff(indices))

    # print(depth,'mm image depth with a pixel:', depth_pix, 'mm')

    return depth_pix


### Get pleura pomponents (from Ale):


class Point:
    x = 0.0
    y = 0.0
    dist = 0.0


def IdnetifyPoly(X_, Y_, order):
    X_ = np.asarray(X_, dtype=np.float64)
    Y_ = np.asarray(Y_, dtype=np.float64)

    valid = np.isfinite(X_) & np.isfinite(Y_)
    X_ = X_[valid]
    Y_ = Y_[valid]

    if len(X_) == 0 or len(Y_) == 0:
        poly_line = np.poly1d([0.0])
        return poly_line, np.array([])

    if len(X_) < 2 or len(np.unique(Y_)) < 2:
        const_x = float(np.median(X_))
        poly_line = np.poly1d([const_x])
        return poly_line, np.ones_like(Y_) * const_x

    safe_order = min(order, len(X_) - 1, len(np.unique(Y_)) - 1)

    try:
        poly_line = np.poly1d(np.polyfit(Y_, X_, safe_order))
        fitted_X = poly_line(Y_)
        return poly_line, fitted_X
    except Exception:
        const_x = float(np.median(X_))
        poly_line = np.poly1d([const_x])
        return poly_line, np.ones_like(Y_) * const_x


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
        y = p.y
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


def build_filters():
    Filters = []
    ksize = 40
    for theta in np.arange(0, np.pi, np.pi / 4):
        params = {'ksize': (ksize, ksize), 'sigma': 3.3, 'theta': theta, 'lambd': 18.3,
                  'gamma': 4.5, 'psi': 0.89, 'ktype': cv2.CV_32F}
        Kern = cv2.getGaborKernel(**params)  # Create the kernel
        Kern /= 1.5 * Kern.sum()
        Filters.append((Kern, params))
    return Filters


def process(img, filters):
    accum = np.zeros_like(img)  # initialize the same size matrix as img
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def ComputeDistance(p1, p2):
    distance = math.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2))
    return distance


def removeOutliers(points, outlierConstant):
    if len(points) < 2:
        return np.zeros((0, 0), dtype=np.uint8)

    matrx = np.zeros(shape=(len(points), len(points)), dtype=np.uint8)

    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                dist = int(ComputeDistance(points[i], points[j]))
                if dist <= outlierConstant:
                    matrx[i][j] = 1

    return matrx


def ExtractContour(Image, points):
    for y in range(0, Image.shape[1] - 1, 1):
        for x in range(Image.shape[0] - 1, 0, -1):
            if Image[x, y] == 1:
                p = Point()
                p.x = x
                p.y = y
                points.append(p)
                break


def RemapIntensity(Image, low, high):
    dst = cv2.bilateralFilter(src=Image, d=0, sigmaColor=20, sigmaSpace=15)
    # remap = rescale_intensity(Image,in_range = (low,high))
    return dst


def to_gray_float(Image):
    Image = np.asarray(Image)

    if Image.ndim == 2:
        gray = Image.astype(np.float32)
    elif Image.ndim == 3:
        if Image.shape[2] == 4:
            Image = cv2.cvtColor(Image, cv2.COLOR_RGBA2RGB)
        gray = rgb2gray(Image).astype(np.float32)
    else:
        raise ValueError("Imagine invalida pentru conversie grayscale: " + str(Image.shape))

    if gray.size > 0 and np.max(gray) > 1.0:
        gray = gray / 255.0

    return gray


def to_uint8_image(image):
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)

    if image.size == 0:
        return image.astype(np.uint8)

    if np.max(image) <= 1.0:
        image = image * 255.0

    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def reduce_palette_to_gray(image, colors):
    image = to_uint8_image(image)

    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L")
    else:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        pil_image = Image.fromarray(image)

    palette_image = pil_image.convert("P", palette=Image.ADAPTIVE, colors=colors)
    gray_image = palette_image.convert("L")

    return to_gray_float(np.array(gray_image))


def Binarize(Image, tresh):
    Image = to_gray_float(Image)

    if Image.size == 0:
        return Image.astype(bool)

    binarized = Image > tresh
    ratio = np.count_nonzero(binarized) / max(binarized.size, 1)

    if ratio < 0.001:
        adaptive_tresh = np.percentile(Image, 92)
        binarized = Image >= adaptive_tresh

    elif ratio > 0.60:
        adaptive_tresh = np.percentile(Image, 85)
        binarized = Image >= adaptive_tresh

    return binarized


def PreprocessImage(Image):
    Image = to_gray_float(Image)
    return Image


def CropImage(Image):
    crop_img, top_offset, left_offset = CropBorderWithOffset(Image)
    return crop_img, top_offset


def GetYellowPix(Image):
    r_query = 169
    g_query = 169
    b_query = 105

    yellow_pixels = []
    yellow_pixels = np.where((Image[:, :, 0] >= r_query) & (Image[:, :, 1] >= g_query) & (Image[:, :, 2] <= b_query))
    return yellow_pixels


def FilterImage(Image):
    uniform_result = ndimage.uniform_filter(Image, size=20)
    return uniform_result


def ExtractConnectedComponents(distances, points):
    X_ = []
    Y_ = []
    pts = []

    if len(points) < 2 or distances is None or distances.size == 0:
        return pts, X_, Y_

    graph = csr_matrix(distances)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    if n_components <= 0 or len(labels) == 0:
        return pts, X_, Y_

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

    arr = np.asarray(ROI)

    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return False

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if arr.dtype == bool:
        img = arr.astype(np.uint8) * 255
    else:
        arr = arr.astype(np.float32)

        if np.max(arr) <= 1.0:
            img = (arr > 0).astype(np.uint8) * 255
        else:
            img = np.clip(arr, 0, 255).astype(np.uint8)

    unique_values = np.unique(img)

    if len(unique_values) <= 2:
        binary = img > 0
    else:
        try:
            thresh = threshold_yen(img) - 0.2 * threshold_yen(img)
        except Exception:
            return False

        binary = img > thresh

    mask = binary.astype(np.uint8)

    if np.count_nonzero(mask) < 2:
        return False

    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]

    if stats.shape[0] <= 1:
        return False

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    new_img = np.zeros_like(mask, dtype=np.uint8)
    new_img[labels == largest_label] = 255

    new_img = cv2.dilate(new_img, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(
        new_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours is None or len(contours) == 0:
        return False

    return contours


def contours_from_mask(mask):
    if mask is None:
        return False

    mask = np.asarray(mask)

    if mask.size == 0:
        return False

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    mask = (mask > 0).astype(np.uint8) * 255

    if np.count_nonzero(mask) < 2:
        return False

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours is None or len(contours) == 0:
        return False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def IdentifyPrincipalContour(img):
    points = list()

    ExtractContour(img, points)

    if len(points) < 2:
        raise ValueError("[IdentifyPrincipalContour] Nu exista suficiente puncte candidate pentru pleura.")

    distances = removeOutliers(points, 50)
    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 2:
        raise ValueError("[IdentifyPrincipalContour] Componenta principala este prea mica.")

    poly_line, PX_ = IdnetifyPoly(PX_, PY_, 3)

    deviation = DOC_TRAVELER_DEVIATION
    pleural_underline = Fit(pts, poly_line, deviation)

    if len(pleural_underline) < 2:
        pleural_underline = pts

    if len(pleural_underline) < 2:
        raise ValueError("[IdentifyPrincipalContour] Nu s-a putut forma underline-ul pleural.")

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
        pts = pleural_underline
        PX = [p.x for p in pts]
        PY = [p.y for p in pts]
    else:
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
            raise ValueError("[IdentifyPrincipalContour] ROI-ul principal nu a produs un contur valid.")

    # Conform documentatiei: dupa cel mai mare contur conectat,
    # calculam o polilinie mai rigida de grad 2 si eliminam artefactele
    # care sunt prea departe de aceasta. Nu modificam artificial punctele
    # ca in Fit2; doar pastram punctele valide.
    rigid_poly_line, _ = IdnetifyPoly(PX, PY, DOC_FINAL_POLY_ORDER)

    deviation = DOC_COMPONENT_DEVIATION
    pleura = Fit(pts, rigid_poly_line, deviation)
    PX_ = [p.x for p in pleura]
    PY_ = [p.y for p in pleura]

    if len(pleura) < 2 or len(PX_) < 2 or len(PY_) < 2:
        # Fallback documentat: daca polinomul rigid elimina prea mult,
        # pastram filtrarea fata de polinomul initial de grad 3, tot cu limita 50.
        pleura = Fit(pts, poly_line, DOC_COMPONENT_DEVIATION)
        PX_ = [p.x for p in pleura]
        PY_ = [p.y for p in pleura]

    if len(pleura) < 2 or len(PX_) < 2 or len(PY_) < 2:
        raise ValueError("[IdentifyPrincipalContour] Nu exista suficiente puncte dupa filtrarea conform documentatiei.")

    poly_line2, PX_fit = IdnetifyPoly(PX_, PY_, DOC_FINAL_POLY_ORDER)

    ps = []
    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    if len(ps) < 2:
        raise ValueError("[IdentifyPrincipalContour] Contur principal invalid.")

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)
    return ctr, PY_, poly_line2, pleura


def shifted_poly_function(local_poly, column_offset):
    def global_poly(y):
        y_arr = np.asarray(y)
        result = local_poly(y_arr - column_offset)

        if np.ndim(result) == 0:
            return float(result)

        return result

    return global_poly


def IdentifyPrincipalContourCentral(img):
    h, w = img.shape[:2]
    last_error = None

    for left_frac, right_frac in PRINCIPAL_SEARCH_BANDS:
        y1 = int(round(left_frac * w))
        y2 = int(round(right_frac * w))

        y1 = max(0, min(y1, w - 2))
        y2 = max(y1 + 2, min(y2, w))

        if y2 - y1 < 20:
            continue

        local_img = img[:, y1:y2].copy()

        try:
            PrincipalComponent, PYS, local_poly_line2, points = IdentifyPrincipalContour(local_img)
        except Exception as exc:
            last_error = exc
            continue

        PrincipalComponent = PrincipalComponent.copy()
        PrincipalComponent[:, :, 0] += y1

        shifted_PYS = [float(y) + y1 for y in PYS]

        for p in points:
            p.y = p.y + y1

        global_poly_line2 = shifted_poly_function(local_poly_line2, y1)

        if DEBUG_VERBOSE:
            print(
                "Banda centrala folosita pentru componenta principala:",
                str(y1) + ":" + str(y2),
                "din",
                w
            )

        return PrincipalComponent, shifted_PYS, global_poly_line2, points

    raise ValueError(
        "[IdentifyPrincipalContourCentral] Nu s-a putut identifica pleura principala in zona centrala. Ultima eroare: "
        + str(last_error)
    )


def IdentifySecondaryContour(lateral_poly, img, minX, minY):
    points = list()

    ExtractContour(img, points)

    if len(points) < 2:
        return False, False, False, False, 1

    distances = removeOutliers(points, 50)
    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 2:
        return False, False, False, False, 1

    poly_line, PX_fit = IdnetifyPoly(PX_, PY_, 3)

    deviation = DOC_TRAVELER_DEVIATION
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

    if len(pts) < 2:
        return False, False, False, False, 1

    # Conform documentatiei, componentele secundare sunt filtrate fata de
    # polilinia originala/laterala si sunt pastrate doar la deviere maxima 50.
    # Nu folosim Fit2 pentru ca acesta muta punctele; documentatia cere eliminarea artefactelor.
    deviation = DOC_COMPONENT_DEVIATION
    pleura = Fit(pts, lateral_poly, deviation)
    PX_ = [p.x for p in pleura]
    PY_ = [p.y for p in pleura]

    if len(pleura) < 2:
        return False, False, False, False, 1

    ps = []
    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)
    return ctr, PY_, PX_, pleura, 0


def contour_basic_stats(component):
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
        "min_col": int(np.min(cols)),
        "max_col": int(np.max(cols)),
        "min_row": int(np.min(rows)),
        "max_row": int(np.max(rows)),
    }


def is_secondary_component_valid(component, principal_poly, image_shape, side="left"):
    """Validare conform pasului 5.3 din documentatie.

    Din toate contururile gasite pastram doar componentele care au deviere
    maxima 50 px fata de polinomul original al componentei principale.
    Nu introducem reguli specifice unei imagini: nu respingem dupa directia
    coborarii, forma locala sau partea stanga/dreapta.
    """
    stats = contour_basic_stats(component)

    if stats is None:
        return False, "invalid_component"

    try:
        expected_rows = principal_poly(stats["cols"])
        max_abs_dev = float(np.max(np.abs(stats["rows"] - expected_rows)))

        if max_abs_dev > DOC_COMPONENT_DEVIATION:
            return False, "deviation_over_50_from_principal_poly"

    except Exception:
        return False, "poly_check_failed"

    return True, "ok"


def MergeLeftComponent(PrincipalComponent, component):
    indxP = 0
    indxC = 0
    i = 0
    Ymin = 3000
    XYmin = 0
    for p in PrincipalComponent:
        for tup in p:
            if tup[0] < Ymin:
                indxP = i
                Ymin = tup[0]
                XYmin = tup[1]
        i += 1

    i = 0
    Ymax = 0
    XYmax = 0
    for p in component:
        for tup in p:
            if tup[0] > Ymax:
                indxC = i
                Ymax = tup[0]
                XYmax = tup[1]
        i += 1

    len1 = len(component)
    indexing = 20
    i1 = GetIndex(indxC, indexing, len1, -1)
    i2 = GetIndex(indxC, indexing, len1, 1)
    P11 = component[GetIndex(indxC, indexing, len1, -1)]
    P51 = component[indxC]
    P01 = component[GetIndex(indxC, indexing, len1, 1)]

    len2 = len(PrincipalComponent)
    i1 = GetIndex(indxP, indexing, len2, -1)
    i2 = GetIndex(indxP, indexing, len2, 1)
    P10 = PrincipalComponent[GetIndex(indxP, indexing, len2, -1)]
    P5 = PrincipalComponent[indxP]
    P0 = PrincipalComponent[GetIndex(indxP, indexing, len2, 1)]

    p1 = P10[0]
    p2 = P5[0]
    p3 = P0[0]

    p4 = P11[0]
    p5 = P51[0]
    p6 = P01[0]
    return p1, p2, p3, p4, p5, p6


def AddContourOnMask(component, mask):
    for j in range(0, mask.shape[0] - 1):
        for i in range(0, mask.shape[1] - 1):
            if cv2.pointPolygonTest(component, (i, j), False) > 0:
                mask[j][i] = 1


def GetIndex(indx, value, length, operator):
    if operator > 0 and indx + value < length - 1:
        indx = indx + value
    if operator < 0 and indx - value > 0:
        indx = indx - value
    if operator > 0 and indx + value > length - 1:
        indx = indx + value - length
    if operator < 0 and indx - value < 0:
        indx = length - (value - indx)
    return indx


def ExtractPleuralLine(orig_Image, interpreted_Image):
    # ASTEA LE-AM COMENTAT EU SI AM INITIALIZAT:
    # img = orig_Image
    # MINx = 0

    ################ Original:

    img, MINx = CropImage(orig_Image)

    orig_Image, MINx = CropImage(orig_Image)
    interpreted_Image, MINx = CropImage(interpreted_Image)

    img = reduce_palette_to_gray(img, colors=DOC_PALETTE_COLORS)
    img = Binarize(img, 0.4)

    if DEBUG_VERBOSE:
        foreground_ratio = np.count_nonzero(img) / max(img.size, 1)
        print("Foreground ratio dupa binarizare principala:", round(float(foreground_ratio), 5))

    left_contour_components = []
    right_contour_components = []
    contour_components = []
    contours = []

    PrincipalComponent, PYS, poly_line2, points = IdentifyPrincipalContour(img)

    contours += points
    cv2.drawContours(interpreted_Image, [PrincipalComponent], 0, (0, 255, 0), 1)

    Ymin = int(min(PYS) + 10)
    Ymin = max(0, min(Ymin, img.shape[1] - 1))
    Ex = []
    Ey = []
    for y in range(0, Ymin):
        Ey.append(y)
        x_hat = poly_line2(y)
        Ex.append(x_hat)

    minX = int(min(Ex)) - 40
    maxX = img.shape[0] - 1
    left_poly_line, Ex = IdnetifyPoly(Ex, Ey, 3)

    newROI = orig_Image[minX:maxX, 0:Ymin]

    last_Y = Ymin + 5
    while Ymin < last_Y:

        newROI = reduce_palette_to_gray(newROI, colors=DOC_PALETTE_COLORS)
        newROI = Binarize(newROI, 0.4)
        if len(newROI) > 0:
            NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(left_poly_line, newROI, minX, 0)
        else:
            Ymin += 1000

        if isEmpty != 1:
            is_valid, reject_reason = is_secondary_component_valid(
                NextComponent,
                poly_line2,
                interpreted_Image.shape[:2],
                side="left"
            )

            if not is_valid:
                if DEBUG_VERBOSE:
                    print("Componenta stanga respinsa:", reject_reason)
                Ymin += 1000
                continue

            contours += points
            cv2.drawContours(interpreted_Image, [NextComponent], 0, (50, 150, 255), 1)
            left_contour_components.append(NextComponent)
            last_Y = Ymin
            Ymin = int(min(PY_) + 10)
            minX = int(min(PX_) - 10)
            newROI = orig_Image[minX:maxX, 0:Ymin]

        else:
            Ymin += 1000

    Ymax = int(max(PYS) - 10)
    Ymax = max(0, min(Ymax, img.shape[1] - 1))
    Ex = []
    Ey = []
    for y in range(Ymax, img.shape[1] - 1):
        Ey.append(y)
        x_hat = poly_line2(y)
        Ex.append(x_hat)

    minX = int(min(Ex)) - 30
    maxX = img.shape[0] - 1
    maxY = int(Ymax)
    right_poly_line, Ex = IdnetifyPoly(Ex, Ey, 3)
    newROI = orig_Image[minX:maxX, Ymax:img.shape[1] - 1]

    last_Y = Ymax - 5
    while Ymax > last_Y:

        newROI = reduce_palette_to_gray(newROI, colors=DOC_PALETTE_COLORS)
        newROI = Binarize(newROI, 0.4)

        if len(newROI) > 0:
            NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(right_poly_line, newROI, minX, maxY)
        else:
            Ymax -= 1000

        if isEmpty != 1:
            is_valid, reject_reason = is_secondary_component_valid(
                NextComponent,
                poly_line2,
                interpreted_Image.shape[:2],
                side="right"
            )

            if not is_valid:
                if DEBUG_VERBOSE:
                    print("Componenta dreapta respinsa:", reject_reason)
                Ymax -= 1000
                continue

            contours += points
            cv2.drawContours(interpreted_Image, [NextComponent], 0, (255, 0, 0), 1)
            right_contour_components.append(NextComponent)
            last_Y = Ymax
            Ymax = int(max(PY_))
            minX = int(min(PX_) - 10)
            newROI = orig_Image[minX:maxX, Ymax:img.shape[1] - 1]
        else:
            Ymax -= 1000

    mask = np.zeros([interpreted_Image.shape[0], interpreted_Image.shape[1]], np.uint8)

    if DEBUG_DRAW_COMPONENTS_SEPARATELY:
        cv2.drawContours(mask, [PrincipalComponent], -1, 255, 2)

        for component in right_contour_components:
            cv2.drawContours(mask, [component], -1, 255, 2)

        for component in left_contour_components:
            cv2.drawContours(mask, [component], -1, 255, 2)

    else:

        if len(right_contour_components) > 0:
            component = right_contour_components[0]

            p1, p2, p3, p4, p5, p6 = MergeLeftComponent(component, PrincipalComponent)
            points = np.array(
                [[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]], [p5[0], p5[1]], [p6[0], p6[1]]])
            hull = cv2.convexHull(points)
            if cv2.contourArea(component) > 400:
                cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
                cv2.drawContours(mask, [component], -1, (255, 255, 255), -1)

            for i in range(0, len(right_contour_components) - 2):
                p1, p2, p3, p4, p5, p6 = MergeLeftComponent(right_contour_components[i + 1],
                                                            right_contour_components[i])
                points = np.array(
                    [[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]], [p5[0], p5[1]], [p6[0], p6[1]]])
                hull = cv2.convexHull(points)
                if cv2.contourArea(right_contour_components[i + 1]) > 400:
                    cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
                    cv2.drawContours(mask, [right_contour_components[i + 1]], -1, (255, 255, 255), -1)

        if len(left_contour_components) > 0:
            component = left_contour_components[0]

            p1, p2, p3, p4, p5, p6 = MergeLeftComponent(PrincipalComponent, component)
            points = np.array(
                [[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]], [p5[0], p5[1]], [p6[0], p6[1]]])
            hull = cv2.convexHull(points)
            if cv2.contourArea(component) > 400:
                cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
                cv2.drawContours(mask, [component], -1, (255, 255, 255), -1)

            for i in range(0, len(left_contour_components) - 2):
                p1, p2, p3, p4, p5, p6 = MergeLeftComponent(left_contour_components[i], left_contour_components[i + 1])
                points = np.array(
                    [[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]], [p5[0], p5[1]], [p6[0], p6[1]]])
                hull = cv2.convexHull(points)
                x = cv2.contourArea(left_contour_components[i + 1])
                if cv2.contourArea(left_contour_components[i + 1]) > 400:
                    cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
                    cv2.drawContours(mask, [left_contour_components[i + 1]], -1, (255, 255, 255), -1)

        cv2.drawContours(mask, [PrincipalComponent], -1, (255, 255, 255), -1)

    cnt = contours_from_mask(mask)

    if cnt is not False:
        ps = []
        for k in cnt:
            for i in k:
                for j in i:
                    ps.append(tuple([int(j[0] + LAST_CROP_LEFT_OFFSET), int(j[1] + MINx)]))
    else:
        ps = []
        for k in PrincipalComponent:
            for j in k:
                ps.append(tuple([int(j[0] + LAST_CROP_LEFT_OFFSET), int(j[1] + MINx)]))

    if len(ps) < 2:
        raise ValueError("[ExtractPleuralLine] Masca finala nu contine contur valid.")

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)

    components = []
    components.append(PrincipalComponent)

    for comp in range(len(right_contour_components)):
        components.append(right_contour_components[comp])

    for comp in range(len(left_contour_components)):
        components.append(left_contour_components[comp])

    component_groups_crop = {
        "principal": [PrincipalComponent],
        "left": left_contour_components,
        "right": right_contour_components,
    }

    component_groups_original = shift_component_groups_to_original(
        component_groups_crop,
        dx=LAST_CROP_LEFT_OFFSET,
        dy=MINx
    )

    return ctr, components, orig_Image, mask, component_groups_crop, component_groups_original


### Get width and irregularity parameters:


def GetVerticals(x, y):
    xall = range(np.min(x), np.max(x))
    yall = range(np.min(y), np.max(y))

    verticals = np.zeros(len(xall), dtype=int)
    vert_mean = np.zeros(len(xall), dtype=int)

    x_plot = range(len(xall) + np.min(x))
    offset = len(x_plot[np.min(x):])

    ii = -1
    maxv = 0
    for iv in xall:
        yindex = np.argwhere(x == iv)
        Y_point = y[yindex]
        ii = ii + 1
        verticals[ii] = np.abs(Y_point[0] - Y_point[-1])
        vert_mean[ii] = np.abs(Y_point[0] + Y_point[-1]) / 2

        if verticals[ii] > maxv:
            maxv = verticals[ii]
            Ymaxv = yindex
            Xmaxv = iv

    return verticals, vert_mean, offset


def PeakValleyDiff(x, y, verticals, vert_mean):
    xall = range(np.min(x), np.max(x))
    yall = range(np.min(y), np.max(y))

    figure(num=None, figsize=(18, 3))
    x_plot = range(len(xall) + np.min(x))
    plt.plot(x, y + np.mean(y))
    plt.plot(x_plot[np.min(x):], verticals, 'r')
    plt.plot(x_plot[np.min(x):], np.zeros(len(verticals)), 'r')

    figure(num=None, figsize=(18, 2))
    x_plot = range(len(xall) + np.min(x))
    plt.plot(x, y, 'lightblue')
    offset = x_plot[np.min(x):]
    plt.plot(offset, vert_mean, 'forestgreen')

    peaks, properties = find_peaks(vert_mean)

    figure(num=None, figsize=(18, 2))
    plt.plot(vert_mean, 'forestgreen')

    plt.plot(peaks, vert_mean[peaks], "r^")

    valleys, properties = find_peaks(-vert_mean)

    figure(num=None, figsize=(18, 2))
    plt.plot(vert_mean, 'forestgreen')

    plt.plot(valleys, vert_mean[valleys], "rv")

    while len(peaks) != len(valleys):
        if len(peaks) > len(valleys):
            valleys = np.append(valleys, peaks[-1])
        else:
            peaks = np.append(peaks, valleys[-1])

    pvs = np.vstack((peaks, valleys))
    pvs = np.transpose(pvs[:])

    Dif = np.abs(np.diff(pvs))
    PVDif = 1 / np.mean(Dif) * 100
    # print('Peaks-Valleys difference', PVDif)

    return PVDif, offset


def Width_and_Irreg(orig_Image, ctr):
    contours = []
    contours.append(ctr)

    # take all point coordinates contained in the contour line
    x = []
    y = []
    for k in contours:
        for i in k:
            for j in i:
                x.append(j[0])
                y.append(j[1])

    x = np.array(x)
    y = np.array(np.max(y) - y)

    # vertical distances between the upper and lower contour
    verticals, vert_mean, offset = GetVerticals(x, y)

    WidthPleuraPixels = np.mean(verticals)
    # print('Pleural width in pixels:', WidthPleuraPixels)
    # print('Pleural max width in pixels:', np.max(verticals))
    PleuraVariation = np.var(verticals)
    # print('Pleural width variance:', PleuraVariation)

    PVDif, offset = PeakValleyDiff(x, y, verticals, vert_mean)

    print('           - - - Pleura Width - - -')
    print('Pleural width in pixels:     ', WidthPleuraPixels)
    print('Pleural max width in pixels: ', np.max(verticals))
    print(' ')
    print('           - - - Pleura Irregularity - - -')
    print('Pleural width variance:      ', PleuraVariation)
    print('Peaks-Valleys difference:    ', PVDif)

    return np.max(y) - vert_mean, offset


### Get interruptions parameters:

# combinate cele doua de mai sus pentru toate componentele dar pe o singura imagine

def Interruptions(imgo, middle_line, offset, components):
    orig_Image = imgo.copy()

    result_mask = []

    for comp in range(len(components)):
        contours = []
        contours.append(components[comp])

        pleura_only = cv2.cvtColor(orig_Image, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(pleura_only)
        cv2.drawContours(mask, contours, 0, 255, -1)

        pleura_only[mask == 0] = 0

        # nu e neaparat nevoie
        out = np.zeros_like(pleura_only)  # Extract out the object and place into output image
        out[mask == 255] = pleura_only[mask == 255]

        interrupted = out.copy()

        thresh = threshold_otsu(interrupted) * 1.7
        # print('Threshold for component',comp,'is:',thresh)

        binary = interrupted > thresh
        interrupted = binary.astype(np.uint8)  # convert to an unsigned byte
        interrupted *= 255

        full_mask = out.copy()
        full_mask[full_mask > 0] = 255

        result_mask.append(interrupted)

    result = np.zeros_like(pleura_only)
    for mask in result_mask:
        result = cv2.add(result, mask)

    # Now crop
    (y, x) = np.where(result.copy() == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    result = result[topy:bottomy, 0:orig_Image.shape[1]]

    # show results
    figure(num=None, figsize=(20, 20))
    plt.imshow(result, cmap='gray')
    plt.plot(offset, middle_line - 2, 'forestgreen')

    interrupted_count = 0
    inter_width = 0
    inter_number = 0

    for i, j in zip(offset, middle_line):
        if result[j][i] == 0:
            interrupted_count = interrupted_count + 1
            inter_width = inter_width + 1
        if result[j][i] != 0:
            if inter_width > 15:  # MANUALY THRESCHOLDED THE INTERRUPTION LENGTH
                inter_number = inter_number + 1
            # print(inter_width)
            inter_width = 0

    interrput_proportion = round(interrupted_count / len(middle_line) * 100, 2)

    print(' ')
    print('           - - - Pleura Interruptions - - -')
    print('There are ', interrupted_count, ' interruption pixels out of ', len(middle_line), ' with a proportion of:',
          interrput_proportion, '%')
    print('Number of interruptions: ', inter_number)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def image_path_from_index(idx):
    candidates = [
        os.path.join(INPUT_DIR, str(idx) + ".jpg"),
        os.path.join(INPUT_DIR, str(idx) + ".jpeg"),
        os.path.join(INPUT_DIR, str(idx) + ".png"),
        os.path.join(INPUT_DIR, str(idx) + ".bmp"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image


def draw_final_contour(image, ctr):
    out = image.copy()

    try:
        cv2.drawContours(out, [ctr], -1, (255, 255, 255), 2)
    except Exception:
        pass

    return out


def build_mask_from_contour(image_shape, ctr, thickness=-1):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    try:
        cv2.drawContours(mask, [ctr], -1, 255, thickness)
    except Exception:
        pass

    return mask


def build_mask_from_components(image_shape, components, thickness=-1):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for component in components:
        try:
            cv2.drawContours(mask, [component], -1, 255, thickness)
        except Exception:
            pass

    return mask


def shift_contour(component, dx=0, dy=0):
    if component is False or component is None:
        return None

    shifted = component.copy().astype(np.int32)
    shifted[:, :, 0] += int(dx)
    shifted[:, :, 1] += int(dy)
    return shifted


def shift_component_groups_to_original(component_groups, dx=0, dy=0):
    shifted_groups = {}

    for name, comps in component_groups.items():
        shifted_groups[name] = []

        for comp in comps:
            shifted = shift_contour(comp, dx=dx, dy=dy)

            if shifted is not None:
                shifted_groups[name].append(shifted)

    return shifted_groups


def build_mask_from_component_groups(image_shape, component_groups, thickness=2):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for comps in component_groups.values():
        for comp in comps:
            try:
                cv2.drawContours(mask, [comp], -1, 255, thickness)
            except Exception:
                pass

    return mask


def draw_component_groups_overlay(image_rgb, component_groups):
    out = image_rgb.copy()

    colors = {
        "principal": (0, 255, 0),
        "left": (255, 165, 0),
        "right": (0, 160, 255),
    }

    for name, comps in component_groups.items():
        color = colors.get(name, (255, 255, 255))

        for comp in comps:
            try:
                cv2.drawContours(out, [comp], -1, color, 2)
            except Exception:
                pass

    return out


def save_component_group_masks(out_dir, idx, image_shape, component_groups, suffix):
    paths = {}

    for name, comps in component_groups.items():
        mask = build_mask_from_components(image_shape, comps, thickness=2)
        path = os.path.join(out_dir, str(idx) + "_" + name + "_components_" + suffix + ".png")
        cv2.imwrite(path, mask)
        paths[name] = path

    return paths


def save_mask_debug_images(idx, orig_image, pleura_image, ctr, components, merged_mask, overlay,
                           component_groups_crop=None, component_groups_original=None):
    out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "MASK_DEBUG"))

    final_mask_filled = build_mask_from_contour(orig_image.shape, ctr, thickness=-1)
    final_mask_outline = build_mask_from_contour(orig_image.shape, ctr, thickness=2)
    components_mask_crop = build_mask_from_components(pleura_image.shape, components, thickness=-1)

    separate_components_mask_crop = None
    separate_components_mask_original = None

    if component_groups_crop is not None:
        separate_components_mask_crop = build_mask_from_component_groups(
            pleura_image.shape,
            component_groups_crop,
            thickness=2
        )

    if component_groups_original is not None:
        separate_components_mask_original = build_mask_from_component_groups(
            orig_image.shape,
            component_groups_original,
            thickness=2
        )

    paths = {
        "overlay": os.path.join(out_dir, str(idx) + "_overlay.png"),
        "final_mask_filled": os.path.join(out_dir, str(idx) + "_final_mask_filled_original.png"),
        "final_mask_outline": os.path.join(out_dir, str(idx) + "_final_mask_outline_original.png"),
        "merged_mask_crop": os.path.join(out_dir, str(idx) + "_merged_mask_crop.png"),
        "components_mask_crop": os.path.join(out_dir, str(idx) + "_components_mask_crop.png"),
        "components_separate_crop": os.path.join(out_dir, str(idx) + "_components_separate_outline_crop.png"),
        "components_separate_original": os.path.join(out_dir, str(idx) + "_components_separate_outline_original.png"),
        "components_separate_color_original": os.path.join(out_dir,
                                                           str(idx) + "_components_separate_color_original.png"),
    }

    cv2.imwrite(paths["overlay"], cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["final_mask_filled"], final_mask_filled)
    cv2.imwrite(paths["final_mask_outline"], final_mask_outline)
    cv2.imwrite(paths["merged_mask_crop"], merged_mask)
    cv2.imwrite(paths["components_mask_crop"], components_mask_crop)

    if separate_components_mask_crop is not None:
        cv2.imwrite(paths["components_separate_crop"], separate_components_mask_crop)

    if separate_components_mask_original is not None:
        cv2.imwrite(paths["components_separate_original"], separate_components_mask_original)

    if component_groups_original is not None:
        separate_color_overlay = draw_component_groups_overlay(orig_image, component_groups_original)
        cv2.imwrite(paths["components_separate_color_original"],
                    cv2.cvtColor(separate_color_overlay, cv2.COLOR_RGB2BGR))
        save_component_group_masks(out_dir, idx, orig_image.shape, component_groups_original, "original")

    if component_groups_crop is not None:
        save_component_group_masks(out_dir, idx, pleura_image.shape, component_groups_crop, "crop")

    print("Masti debug salvate in:", out_dir)
    print(" -", paths["final_mask_filled"])
    print(" -", paths["final_mask_outline"])
    print(" -", paths["merged_mask_crop"])
    print(" -", paths["components_mask_crop"])
    print(" -", paths["components_separate_crop"])
    print(" -", paths["components_separate_original"])
    print(" -", paths["components_separate_color_original"])

    return paths, final_mask_filled, final_mask_outline, components_mask_crop


def run_one_image(idx, show_result=True, save_result=False):
    img_path = image_path_from_index(idx)
    img_name = os.path.basename(img_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError("Imaginea nu exista: " + img_path)

    print("Procesez: " + img_name)

    orig_Image = load_image(img_path)

    if ENABLE_NOISE_REMOVAL:
        processing_Image = NoiseRemoval(orig_Image)
        if DEBUG_VERBOSE:
            print("Noise removal aplicat: resize", NOISE_REMOVAL_SIZE, "->", orig_Image.shape[:2])
    else:
        processing_Image = orig_Image.copy()

    interpreted_Image = processing_Image.copy()

    try:
        one_pixel = PixelConverter(orig_Image)
        print("One pixel is:", one_pixel, "mm")
    except Exception:
        one_pixel = None
        print("Nu am putut calcula conversia pixel-mm.")

    result = ExtractPleuralLine(processing_Image, interpreted_Image)

    if len(result) == 6:
        ctr, components, pleura_image, merged_mask, component_groups_crop, component_groups_original = result
    else:
        ctr, components, pleura_image, merged_mask = result
        component_groups_crop = None
        component_groups_original = None

    try:
        x, y, w, h = cv2.boundingRect(ctr)
        print("Contur final: width=", w, "height=", h, "components=", len(components))
    except Exception:
        pass

    if DEBUG_DRAW_COMPONENTS_SEPARATELY and component_groups_original is not None:
        overlay = draw_component_groups_overlay(orig_Image, component_groups_original)
    else:
        overlay = draw_final_contour(orig_Image, ctr)

    mask_paths = None
    final_mask_filled = None
    final_mask_outline = None
    components_mask_crop = None

    if SAVE_DEBUG_MASKS:
        mask_paths, final_mask_filled, final_mask_outline, components_mask_crop = save_mask_debug_images(
            idx,
            orig_Image,
            pleura_image,
            ctr,
            components,
            merged_mask,
            overlay,
            component_groups_crop=component_groups_crop,
            component_groups_original=component_groups_original
        )

    if save_result:
        out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "CONTOURS"))
        out_path = os.path.join(out_dir, str(idx) + "_contour.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if SHOW_WIDTH_AND_INTERRUPTION_PLOTS:
        try:
            middle_line, offset = Width_and_Irreg(pleura_image, ctr)
            Interruptions(pleura_image, middle_line, offset, components)
        except Exception as e:
            print("Analiza width/interruptions a esuat:", str(e))

    if show_result:
        plt.figure(figsize=(12, 7))
        plt.imshow(overlay)
        plt.title("Contur pleural - " + img_name)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        if SHOW_DEBUG_MASKS:
            plt.figure(figsize=(16, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(overlay)
            plt.title("overlay componente separate" if DEBUG_DRAW_COMPONENTS_SEPARATELY else "overlay contur")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            if final_mask_filled is not None:
                plt.imshow(final_mask_filled, cmap="gray")
            else:
                plt.imshow(build_mask_from_contour(orig_Image.shape, ctr, thickness=-1), cmap="gray")
            plt.title("masca finala umpluta - original")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(merged_mask, cmap="gray")
            plt.title(
                "masca interna fara unire - crop" if DEBUG_DRAW_COMPONENTS_SEPARATELY else "masca interna merged - crop")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            if components_mask_crop is not None:
                plt.imshow(components_mask_crop, cmap="gray")
            else:
                plt.imshow(build_mask_from_components(pleura_image.shape, components, thickness=-1), cmap="gray")
            plt.title("masca componente separate - crop")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    return {
        "image": img_name,
        "path": img_path,
        "contour": ctr,
        "components": components,
        "one_pixel": one_pixel,
        "mask_paths": mask_paths,
    }


def main1():
    ensure_dir(OUTPUT_ROOT)

    total = END_IDX - START_IDX + 1
    success = 0
    failed = 0

    for pos, idx in enumerate(range(START_IDX, END_IDX + 1), start=1):
        print("[" + str(pos) + "/" + str(total) + "]")

        try:
            run_one_image(
                idx,
                show_result=False,
                save_result=SAVE_BATCH_RESULTS,
            )
            success += 1
        except Exception as e:
            failed += 1
            print("Eroare la imaginea", idx, ":", str(e))

            err_dir = ensure_dir(os.path.join(OUTPUT_ROOT, "ERRORS"))
            err_path = os.path.join(err_dir, str(idx) + "_error.txt")

            with open(err_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    print("\nFinalizat.")
    print("Reusite:", success)
    print("Esuate:", failed)


def main2():
    run_one_image(
        SINGLE_IMAGE_IDX,
        show_result=SHOW_SINGLE_RESULT,
        save_result=False,
    )


if __name__ == "__main__":
    if RUN_SINGLE_IMAGE:
        main2()
    else:
        main1()
