import math
import os
import re
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tes
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage import img_as_ubyte, io
from skimage.filters import threshold_yen


# ============================================================
# CONFIG
# ============================================================

IMG_PATH = r'C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES\5.jpg'
DEBUG_DIR = r'C:\Facultate\AN4\Licenta\Licenta-Cod\DEBUG_OUT'

ONE_PIXEL_FALLBACK = 0.07

MAX_SIDE_COMPONENT_SLOPE = 0.22
MAX_SIDE_COMPONENT_HEIGHT = 35
MAX_SIDE_COMPONENT_DEVIATION = 30

FINAL_MASK_MAX_DEVIATION = 28
FINAL_MASK_MAX_COLUMN_HEIGHT = 40

RIGHT_EXTENSION_ENABLED = True
RIGHT_EXTENSION_MIN_MISSING_WIDTH = 25
RIGHT_EXTENSION_SEARCH_BAND_PX = 18
RIGHT_EXTENSION_MAX_STEP_PX = 4
RIGHT_EXTENSION_MAX_MISSES = 14
RIGHT_EXTENSION_MIN_POINTS = 12
RIGHT_EXTENSION_SCORE_PERCENTILE = 70
RIGHT_EXTENSION_MIN_HALF_THICKNESS_PX = 2
RIGHT_EXTENSION_MAX_HALF_THICKNESS_PX = 7

CONTINUOUS_CONTOUR_ENABLED = True
CONTINUOUS_MAX_GAP_PX = 120
CONTINUOUS_MAX_CENTER_DIFF_PX = 45
CONTINUOUS_SMOOTH_WINDOW = 9
CONTINUOUS_MIN_WIDTH_PX = 3


# ============================================================
# CROP IMAGE BORDER
# ============================================================

def CropBorder(orig_Image):
    orig_img = orig_Image.copy()

    if orig_img.ndim == 2:
        gray_Image = orig_img.copy()
    else:
        gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

    _, img_bin = cv2.threshold(
        gray_Image,
        128,
        255,
        cv2.THRESH_BINARY | cv2.ADAPTIVE_THRESH_MEAN_C
    )

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
        left_bound = last_small[-1]

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
        print("[CropBorder] Bara verticala nu a fost gasita. Returnez imaginea grayscale.")
        return gray_Image.copy()

    bar_pos = np.where(columns == np.max(columns))[0][0]

    bar = bar_image[:, bar_pos] // 255
    bar_pixels = np.array(np.where(bar == 1))

    if bar_pixels.shape[1] == 0:
        print("[CropBorder] Tick-urile de pe bara nu au fost gasite. Returnez imaginea grayscale.")
        return gray_Image.copy()

    first_bar_pixel = bar_pixels[:, 0]
    last_bar_pixel = bar_pixels[:, -1]

    if first_bar_pixel[0] == 0 or last_bar_pixel[0] == 0:
        print("[CropBorder] Imagine neconforma, nu s-a cropat.")
        return gray_Image.copy()

    x2 = bar_pos - 20

    if x2 <= left_bound:
        print("[CropBorder] Coordonate crop invalide. Returnez imaginea grayscale.")
        return gray_Image.copy()

    crop_img = gray_Image[
        first_bar_pixel[0]:last_bar_pixel[0],
        left_bound:x2
    ].copy()

    if crop_img.size == 0:
        print("[CropBorder] Crop gol. Returnez imaginea grayscale.")
        return gray_Image.copy()

    return crop_img


# ============================================================
# PIXEL CONVERTER
# ============================================================

def PixelConverter(orig_Image):
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
        print("[PixelConverter] Nu am gasit bara. Fallback 0.07.")
        return ONE_PIXEL_FALLBACK

    bar_pos = np.where(columns == np.max(columns))[0][0]

    bar = horizontal_lines_img[:, bar_pos] // 255
    indices = [i for i, x in enumerate(bar) if x == 1]

    if len(indices) < 2:
        print("[PixelConverter] Prea putine tick-uri. Fallback 0.07.")
        return ONE_PIXEL_FALLBACK

    try:
        tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        image_str = tes.image_to_string(np.invert(img_bin))
    except Exception as e:
        print("[PixelConverter] OCR esuat:", e)
        return ONE_PIXEL_FALLBACK

    depth = None

    if 'cm' in image_str:
        depth_str = image_str[image_str.find('cm') - 5:image_str.find('cm')]
        depth_no = [int(i) for i in depth_str if i.isdigit()]
        if len(depth_no) > 0:
            depth = depth_no[0] * 10

    if depth is None and 'mm' in image_str:
        depth_str = image_str[image_str.find('mm') - 5:image_str.find('mm')]
        depth_no = [int(s) for s in re.findall(r'\b\d+\b', depth_str)]
        if len(depth_no) > 0:
            depth = depth_no[0]

    if depth is None:
        if '\nD ' in image_str:
            depth_str = image_str[
                image_str.find('\nD '):image_str.find('\nD ') + 6
            ]
            depth_no = [int(i) for i in depth_str if i.isdigit()]
            if len(depth_no) > 0:
                if depth_no[0] < 50:
                    depth = depth_no[0] * 10
                else:
                    depth = depth_no[0]

        if depth is None and '-D ' in image_str:
            depth_str = image_str[
                image_str.find('-D '):image_str.find('-D ') + 6
            ]
            depth_no = [int(i) for i in depth_str if i.isdigit()]
            if len(depth_no) > 0:
                if depth_no[0] < 50:
                    depth = depth_no[0] * 10
                else:
                    depth = depth_no[0]

    if depth is None:
        print("[PixelConverter] Adancime negasita. Fallback 0.07.")
        return ONE_PIXEL_FALLBACK

    tick_sum = np.sum(np.diff(indices))

    if tick_sum <= 0:
        print("[PixelConverter] Tick sum invalid. Fallback 0.07.")
        return ONE_PIXEL_FALLBACK

    depth_pix = depth / tick_sum

    if depth_pix <= 0 or depth_pix > 0.5:
        print("[PixelConverter] Valoare suspecta. Fallback 0.07.")
        return ONE_PIXEL_FALLBACK

    return depth_pix


# ============================================================
# BASIC STRUCTURES
# ============================================================

class Point:
    x = 0.0
    y = 0.0
    dist = 0.0


def IdnetifyPoly(X_, Y_, order):
    if len(X_) < 2 or len(Y_) < 2:
        p = np.poly1d([0])
        return p, np.array(X_)

    safe_order = min(order, len(X_) - 1)

    p30 = np.poly1d(np.polyfit(Y_, X_, safe_order))
    X_ = p30(Y_)
    return p30, X_


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
    if len(points) == 0:
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
    for y in range(0, Image.shape[1] - 1, 1):
        for x in range(Image.shape[0] - 1, 0, -1):
            if Image[x, y] == 1:
                p = Point()
                p.x = x
                p.y = y
                points.append(p)
                break


def PreparePleuraBinary(
    img,
    top_ratio=0.08,
    bottom_ratio=0.65,
    percentile=88,
    clahe_clip=2.0,
    kernel_h_width=21,
    kernel_v_height=25
):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    h, w = img.shape

    den = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enh = clahe.apply(den)

    grad = cv2.Sobel(enh, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(grad)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    grad = grad.astype(np.uint8)

    score = cv2.addWeighted(enh, 0.60, grad, 0.40, 0)

    y1 = int(top_ratio * h)
    y2 = int(bottom_ratio * h)

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    roi = np.zeros_like(score)
    roi[y1:y2, :] = score[y1:y2, :]

    valid = roi[roi > 0]

    if len(valid) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    th = np.percentile(valid, percentile)

    mask = (roi > th).astype(np.uint8) * 255

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h_width, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=1)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, kernel_v_height))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
    mask = cv2.subtract(mask, vertical)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=1)

    return (mask > 0).astype(np.uint8)


def ToGrayUint8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    return img


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def ExtractConnectedComponents(distances, points):
    X_ = []
    Y_ = []
    pts = []

    if len(points) == 0 or distances.size == 0:
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


def IdentifyPrincipalContour(img):
    points = []

    ExtractContour(img, points)

    if len(points) < 2:
        raise ValueError("[IdentifyPrincipalContour] Prea putine puncte candidate.")

    distances = removeOutliers(points, 50)

    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 2:
        raise ValueError("[IdentifyPrincipalContour] Componenta principala prea mica.")

    poly_line, PX_ = IdnetifyPoly(PX_, PY_, 3)

    deviation = 20
    pleural_underline = Fit(pts, poly_line, deviation)

    if len(pleural_underline) == 0:
        pleural_underline = pts

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

    poly_line2, PX_ = IdnetifyPoly(PX_, PY_, 1)

    ps = []

    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)

    return ctr, PY_, poly_line2, pleura


def IdentifySecondaryContour(lateral_poly, img, minX, minY):
    points = []

    ExtractContour(img, points)

    if len(points) < 2:
        return False, False, False, False, 1

    distances = removeOutliers(points, 50)

    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 2:
        return False, False, False, False, 1

    poly_line, PX_ = IdnetifyPoly(PX_, PY_, 3)

    pleural_underline = Fit(pts, poly_line, 30)

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

    pleura, PX_, PY_ = Fit2(pts, lateral_poly, 100, PX, PY)

    ps = []

    for p in pleura:
        ps.append(tuple([int(p.y), int(p.x)]))

    if len(pleura) > 0:
        ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)
        return ctr, PY_, PX_, pleura, 0

    return False, False, False, False, 1


# ============================================================
# COMPONENT FILTERS
# ============================================================

def IsPleuraLikeComponent(
    component,
    poly_line=None,
    max_slope=MAX_SIDE_COMPONENT_SLOPE,
    max_height=MAX_SIDE_COMPONENT_HEIGHT,
    max_deviation=MAX_SIDE_COMPONENT_DEVIATION
):
    if component is False or component is None:
        return False

    pts = component.reshape(-1, 2)

    if len(pts) < 5:
        return False

    xs = pts[:, 0]
    ys = pts[:, 1]

    width = xs.max() - xs.min()
    height = ys.max() - ys.min()

    if width < 10:
        return False

    if height > max_height:
        return False

    if height / max(width, 1) > 0.35:
        return False

    try:
        p = np.poly1d(np.polyfit(xs, ys, 1))
        slope = abs(p.coefficients[0])
    except Exception:
        return False

    if slope > max_slope:
        return False

    if poly_line is not None:
        try:
            expected_y = poly_line(xs)
            deviation = np.median(np.abs(ys - expected_y))

            if deviation > max_deviation:
                return False
        except Exception:
            return False

    return True


def GetIndex(indx, value, length, operator):
    if length <= 0:
        return 0

    if operator > 0 and indx + value < length - 1:
        indx = indx + value

    if operator < 0 and indx - value > 0:
        indx = indx - value

    if operator > 0 and indx + value > length - 1:
        indx = indx + value - length

    if operator < 0 and indx - value < 0:
        indx = length - (value - indx)

    indx = max(0, min(indx, length - 1))

    return indx


def MergeLeftComponent(PrincipalComponent, component):
    indxP = 0
    indxC = 0

    i = 0
    Ymin = 3000

    for p in PrincipalComponent:
        for tup in p:
            if tup[0] < Ymin:
                indxP = i
                Ymin = tup[0]
        i += 1

    i = 0
    Ymax = 0

    for p in component:
        for tup in p:
            if tup[0] > Ymax:
                indxC = i
                Ymax = tup[0]
        i += 1

    len1 = len(component)
    indexing = 20

    P11 = component[GetIndex(indxC, indexing, len1, -1)]
    P51 = component[indxC]
    P01 = component[GetIndex(indxC, indexing, len1, 1)]

    len2 = len(PrincipalComponent)

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


# ============================================================
# RIGHT EXTENSION
# ============================================================

def GetMaxContourX(components):
    max_x = 0

    for comp in components:
        if comp is False or comp is None:
            continue

        pts = comp.reshape(-1, 2)

        if len(pts) == 0:
            continue

        max_x = max(max_x, int(np.max(pts[:, 0])))

    return max_x


def EstimatePleuraHalfThicknessPx(components):
    heights = []

    for comp in components:
        if comp is False or comp is None:
            continue

        pts = comp.reshape(-1, 2)

        if len(pts) < 5:
            continue

        xs = pts[:, 0]
        ys = pts[:, 1]

        for x in np.unique(xs):
            y_vals = ys[xs == x]

            if len(y_vals) < 2:
                continue

            height = int(np.max(y_vals) - np.min(y_vals))

            if 1 <= height <= 20:
                heights.append(height)

    if len(heights) == 0:
        return 3

    half = int(round(np.median(heights) / 2.0))

    half = max(RIGHT_EXTENSION_MIN_HALF_THICKNESS_PX, half)
    half = min(RIGHT_EXTENSION_MAX_HALF_THICKNESS_PX, half)

    return half


def BuildBandContourFromCenterline(center_pts, half_thickness, image_height=None):
    if center_pts is None or len(center_pts) < 2:
        return None

    center_pts = np.asarray(center_pts, dtype=np.int32)

    top = []
    bottom = []

    for x, y in center_pts:
        y_top = int(y - half_thickness)
        y_bottom = int(y + half_thickness)

        if image_height is not None:
            y_top = max(0, min(image_height - 1, y_top))
            y_bottom = max(0, min(image_height - 1, y_bottom))

        top.append([int(x), y_top])
        bottom.append([int(x), y_bottom])

    top = np.array(top, dtype=np.int32)
    bottom = np.array(bottom, dtype=np.int32)

    contour_pts = np.vstack([top, bottom[::-1]])

    contour = contour_pts.reshape(-1, 1, 2).astype(np.int32)

    return contour


def ExtendRightPleuraByPrediction(orig_gray, poly_line2, base_components):
    if not RIGHT_EXTENSION_ENABLED:
        return None

    gray = ToGrayUint8(orig_gray)
    h, w = gray.shape

    start_x = GetMaxContourX(base_components)

    if w - start_x < RIGHT_EXTENSION_MIN_MISSING_WIDTH:
        return None

    half_thickness = EstimatePleuraHalfThicknessPx(base_components)

    den = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)

    grad = cv2.Sobel(enh, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(grad)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    grad = grad.astype(np.uint8)

    score = cv2.addWeighted(enh, 0.65, grad, 0.35, 0)

    valid_scores = score[score > 0]

    if len(valid_scores) == 0:
        return None

    score_thr = np.percentile(valid_scores, RIGHT_EXTENSION_SCORE_PERCENTILE)

    center_pts = []

    try:
        last_y = int(poly_line2(start_x))
    except Exception:
        last_y = h // 2

    misses = 0

    for x in range(start_x + 1, w):
        try:
            predicted_y = int(poly_line2(x))
        except Exception:
            predicted_y = last_y

        center_y = int(0.7 * predicted_y + 0.3 * last_y)

        y1 = max(0, center_y - RIGHT_EXTENSION_SEARCH_BAND_PX)
        y2 = min(h, center_y + RIGHT_EXTENSION_SEARCH_BAND_PX + 1)

        if y2 <= y1:
            misses += 1

            if misses > RIGHT_EXTENSION_MAX_MISSES:
                break

            continue

        col_score = score[y1:y2, x]

        if len(col_score) == 0:
            misses += 1

            if misses > RIGHT_EXTENSION_MAX_MISSES:
                break

            continue

        best_local = int(np.argmax(col_score))
        best_y = y1 + best_local
        best_score = col_score[best_local]

        if best_score < score_thr:
            misses += 1

            if misses > RIGHT_EXTENSION_MAX_MISSES:
                break

            continue

        misses = 0

        if abs(best_y - last_y) > RIGHT_EXTENSION_MAX_STEP_PX:
            if best_y > last_y:
                best_y = last_y + RIGHT_EXTENSION_MAX_STEP_PX
            else:
                best_y = last_y - RIGHT_EXTENSION_MAX_STEP_PX

        center_pts.append([x, best_y])
        last_y = best_y

    if len(center_pts) < RIGHT_EXTENSION_MIN_POINTS:
        return None

    center_pts = np.array(center_pts, dtype=np.int32)

    if len(center_pts) >= 7:
        ys = center_pts[:, 1].copy()
        ys_smooth = ys.copy()

        for i in range(len(ys)):
            lo = max(0, i - 3)
            hi = min(len(ys), i + 4)
            ys_smooth[i] = int(np.median(ys[lo:hi]))

        center_pts[:, 1] = ys_smooth

    extension_contour = BuildBandContourFromCenterline(
        center_pts,
        half_thickness,
        image_height=h
    )

    return extension_contour


# ============================================================
# CONTINUOUS CONTOUR BUILDER
# ============================================================

def component_to_columns(component):
    if component is False or component is None:
        return {}

    pts = component.reshape(-1, 2)

    if len(pts) < 2:
        return {}

    xs = pts[:, 0]
    ys = pts[:, 1]

    out = {}

    for x in np.unique(xs):
        y_vals = ys[xs == x]

        if len(y_vals) == 0:
            continue

        top = int(np.min(y_vals))
        bottom = int(np.max(y_vals))

        if bottom - top < CONTINUOUS_MIN_WIDTH_PX:
            mid = int(round((top + bottom) / 2.0))
            top = mid - CONTINUOUS_MIN_WIDTH_PX // 2
            bottom = mid + CONTINUOUS_MIN_WIDTH_PX // 2

        out[int(x)] = {
            "top": top,
            "bottom": bottom,
            "center": int(round((top + bottom) / 2.0))
        }

    return out


def smooth_array_median(values, window):
    values = np.asarray(values, dtype=np.float32)

    if len(values) < 3:
        return values

    if window < 3:
        return values

    if window % 2 == 0:
        window += 1

    half = window // 2
    out = values.copy()

    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.median(values[lo:hi])

    return out


def BuildContinuousContourFromComponents(components, image_shape):
    if not CONTINUOUS_CONTOUR_ENABLED:
        return None

    h, w = image_shape[:2]

    comp_infos = []

    for comp in components:
        cols = component_to_columns(comp)

        if len(cols) < 2:
            continue

        xs = sorted(cols.keys())

        comp_infos.append({
            "x_min": xs[0],
            "x_max": xs[-1],
            "cols": cols
        })

    if len(comp_infos) == 0:
        return None

    comp_infos = sorted(comp_infos, key=lambda d: d["x_min"])

    global_cols = {}

    for info in comp_infos:
        for x, data in info["cols"].items():
            if x not in global_cols:
                global_cols[x] = data
            else:
                old = global_cols[x]
                global_cols[x] = {
                    "top": int(round((old["top"] + data["top"]) / 2)),
                    "bottom": int(round((old["bottom"] + data["bottom"]) / 2)),
                    "center": int(round((old["center"] + data["center"]) / 2))
                }

    for i in range(len(comp_infos) - 1):
        left = comp_infos[i]
        right = comp_infos[i + 1]

        x1 = left["x_max"]
        x2 = right["x_min"]

        gap = x2 - x1

        if gap <= 1:
            continue

        if gap > CONTINUOUS_MAX_GAP_PX:
            print(f"[CONTINUOUS] Gap prea mare, nu conectez: {x1}->{x2}, gap={gap}")
            continue

        left_data = left["cols"][x1]
        right_data = right["cols"][x2]

        center_diff = abs(right_data["center"] - left_data["center"])

        if center_diff > CONTINUOUS_MAX_CENTER_DIFF_PX:
            print(
                f"[CONTINUOUS] Diferenta verticala prea mare, nu conectez: "
                f"{x1}->{x2}, diff={center_diff}"
            )
            continue

        xs_gap = np.arange(x1, x2 + 1)

        top_interp = np.linspace(left_data["top"], right_data["top"], len(xs_gap))
        bottom_interp = np.linspace(left_data["bottom"], right_data["bottom"], len(xs_gap))

        for x, yt, yb in zip(xs_gap, top_interp, bottom_interp):
            yt = int(round(yt))
            yb = int(round(yb))

            if yb - yt < CONTINUOUS_MIN_WIDTH_PX:
                mid = int(round((yt + yb) / 2.0))
                yt = mid - CONTINUOUS_MIN_WIDTH_PX // 2
                yb = mid + CONTINUOUS_MIN_WIDTH_PX // 2

            yt = max(0, min(h - 1, yt))
            yb = max(0, min(h - 1, yb))

            global_cols[int(x)] = {
                "top": yt,
                "bottom": yb,
                "center": int(round((yt + yb) / 2.0))
            }

        print(f"[CONTINUOUS] Conectez prin interpolare: {x1}->{x2}, gap={gap}")

    if len(global_cols) < 2:
        return None

    xs = np.array(sorted(global_cols.keys()), dtype=np.int32)

    tops = np.array([global_cols[int(x)]["top"] for x in xs], dtype=np.float32)
    bottoms = np.array([global_cols[int(x)]["bottom"] for x in xs], dtype=np.float32)

    tops = smooth_array_median(tops, CONTINUOUS_SMOOTH_WINDOW)
    bottoms = smooth_array_median(bottoms, CONTINUOUS_SMOOTH_WINDOW)

    top_pts = []
    bottom_pts = []

    for x, yt, yb in zip(xs, tops, bottoms):
        yt = int(round(yt))
        yb = int(round(yb))

        if yb - yt < CONTINUOUS_MIN_WIDTH_PX:
            mid = int(round((yt + yb) / 2.0))
            yt = mid - CONTINUOUS_MIN_WIDTH_PX // 2
            yb = mid + CONTINUOUS_MIN_WIDTH_PX // 2

        yt = max(0, min(h - 1, yt))
        yb = max(0, min(h - 1, yb))

        top_pts.append([int(x), yt])
        bottom_pts.append([int(x), yb])

    top_pts = np.array(top_pts, dtype=np.int32)
    bottom_pts = np.array(bottom_pts, dtype=np.int32)

    contour_pts = np.vstack([top_pts, bottom_pts[::-1]])
    contour = contour_pts.reshape(-1, 1, 2).astype(np.int32)

    return contour


# ============================================================
# OUTPUT HELPERS
# ============================================================

def component_to_open_band_edges(component):
    if component is False or component is None:
        return None, None

    pts = component.reshape(-1, 2)

    if len(pts) < 2:
        return None, None

    xs = pts[:, 0]
    ys = pts[:, 1]

    unique_x = np.unique(xs)

    top_pts = []
    bottom_pts = []

    for x in unique_x:
        y_vals = ys[xs == x]

        if len(y_vals) == 0:
            continue

        y_top = int(np.min(y_vals))
        y_bottom = int(np.max(y_vals))

        top_pts.append([int(x), y_top])
        bottom_pts.append([int(x), y_bottom])

    if len(top_pts) < 2:
        return None, None

    top_pts = np.array(top_pts, dtype=np.int32).reshape(-1, 1, 2)
    bottom_pts = np.array(bottom_pts, dtype=np.int32).reshape(-1, 1, 2)

    return top_pts, bottom_pts


def draw_components_as_open_lines(image, components, color, thickness=2):
    out = image.copy()

    for comp in components:
        top_line, bottom_line = component_to_open_band_edges(comp)

        if top_line is None or bottom_line is None:
            continue

        cv2.polylines(out, [top_line], isClosed=False, color=color, thickness=thickness)
        cv2.polylines(out, [bottom_line], isClosed=False, color=color, thickness=thickness)

    return out


def build_open_line_mask(shape, components, thickness=2):
    mask = np.zeros(shape, dtype=np.uint8)

    for comp in components:
        top_line, bottom_line = component_to_open_band_edges(comp)

        if top_line is None or bottom_line is None:
            continue

        cv2.polylines(mask, [top_line], isClosed=False, color=255, thickness=thickness)
        cv2.polylines(mask, [bottom_line], isClosed=False, color=255, thickness=thickness)

    return mask


# ============================================================
# AUTO SCORE
# ============================================================

def ScoreContourQuality(ctr, image_shape):
    if ctr is None or ctr is False:
        return -1e9

    try:
        pts = ctr.reshape(-1, 2)
    except Exception:
        return -1e9

    if len(pts) < 20:
        return -1e9

    h, w = image_shape[:2]

    xs = pts[:, 0]
    ys = pts[:, 1]

    if len(xs) == 0 or len(ys) == 0:
        return -1e9

    x_min = int(np.min(xs))
    x_max = int(np.max(xs))
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))

    x_span = x_max - x_min
    y_span = y_max - y_min

    coverage = x_span / max(w, 1)

    if coverage < 0.15:
        return -1e9

    unique_x = np.unique(xs)

    thicknesses = []
    centers = []

    for x in unique_x:
        y_vals = ys[xs == x]

        if len(y_vals) < 2:
            continue

        thickness = int(np.max(y_vals) - np.min(y_vals))
        center = int((np.max(y_vals) + np.min(y_vals)) / 2)

        thicknesses.append(thickness)
        centers.append(center)

    if len(thicknesses) < 10:
        return -1e9

    thicknesses = np.array(thicknesses)
    centers = np.array(centers)

    median_thickness = np.median(thicknesses)
    thickness_var = np.var(thicknesses)
    median_center = np.median(centers)

    score = 0.0

    score += coverage * 100.0

    if median_thickness < 2:
        score -= 20
    elif 2 <= median_thickness <= 18:
        score += 30
    elif median_thickness <= 35:
        score += 5
    else:
        score -= 60

    score -= min(thickness_var / 10.0, 80)

    center_ratio = median_center / max(h, 1)

    if center_ratio < 0.03:
        score -= 20

    if center_ratio > 0.75:
        score -= 50

    if y_span > 0.35 * h:
        score -= 50

    return score


# ============================================================
# EXTRACT PLEURAL LINE
# ============================================================

def ExtractPleuralLine(
    orig_Image,
    interpreted_Image,
    prep_percentile=88,
    prep_top_ratio=0.08,
    prep_bottom_ratio=0.65,
    prep_clahe_clip=2.0,
    prep_kernel_h_width=21,
    prep_kernel_v_height=25
):
    img = orig_Image.copy()

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    orig_Image = img.copy()

    if interpreted_Image.ndim == 2:
        interpreted_Image = cv2.cvtColor(interpreted_Image, cv2.COLOR_GRAY2RGB)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    img = PreparePleuraBinary(
        img,
        top_ratio=prep_top_ratio,
        bottom_ratio=prep_bottom_ratio,
        percentile=prep_percentile,
        clahe_clip=prep_clahe_clip,
        kernel_h_width=prep_kernel_h_width,
        kernel_v_height=prep_kernel_v_height
    )

    left_contour_components = []
    right_contour_components = []

    PrincipalComponent, PYS, poly_line2, points = IdentifyPrincipalContour(img)

    cv2.drawContours(interpreted_Image, [PrincipalComponent], 0, (0, 255, 0), 1)

    # -------------------------
    # LEFT SIDE
    # -------------------------

    Ymin = min(PYS) + 10

    Ex = []
    Ey = []

    for y in range(0, Ymin):
        Ey.append(y)
        Ex.append(poly_line2(y))

    minX = int(min(Ex)) - 40
    minX = max(0, minX)

    maxX = img.shape[0] - 1

    left_poly_line, Ex = IdnetifyPoly(Ex, Ey, 3)

    newROI = orig_Image[minX:maxX, 0:Ymin]

    last_Y = Ymin + 5

    while Ymin < last_Y and newROI.shape[1] > 10:
        newROI_proc = PreparePleuraBinary(
            newROI,
            top_ratio=0.0,
            bottom_ratio=1.0,
            percentile=prep_percentile,
            clahe_clip=prep_clahe_clip,
            kernel_h_width=prep_kernel_h_width,
            kernel_v_height=prep_kernel_v_height
        )

        if len(newROI_proc) > 0:
            NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(
                left_poly_line,
                newROI_proc,
                minX,
                0
            )
        else:
            break

        if isEmpty != 1:
            if IsPleuraLikeComponent(NextComponent, poly_line=poly_line2):
                cv2.drawContours(interpreted_Image, [NextComponent], 0, (50, 150, 255), 1)
                left_contour_components.append(NextComponent)

                last_Y = Ymin
                Ymin = int(min(PY_) + 10)
                minX = int(min(PX_) - 10)
                minX = max(0, minX)

                newROI = orig_Image[minX:maxX, 0:Ymin]
            else:
                print("[LEFT] Componenta respinsa.")
                break
        else:
            break

    # -------------------------
    # RIGHT SIDE
    # -------------------------

    Ymax = max(PYS) - 10

    Ex = []
    Ey = []

    for y in range(Ymax, img.shape[1] - 1):
        Ey.append(y)
        Ex.append(poly_line2(y))

    minX = int(min(Ex)) - 30
    minX = max(0, minX)

    maxX = img.shape[0] - 1
    maxY = Ymax

    right_poly_line, Ex = IdnetifyPoly(Ex, Ey, 3)

    newROI = orig_Image[minX:maxX, Ymax:img.shape[1] - 1]

    last_Y = Ymax - 5

    while Ymax > last_Y and newROI.shape[1] > 10:
        newROI_proc = PreparePleuraBinary(
            newROI,
            top_ratio=0.0,
            bottom_ratio=1.0,
            percentile=prep_percentile,
            clahe_clip=prep_clahe_clip,
            kernel_h_width=prep_kernel_h_width,
            kernel_v_height=prep_kernel_v_height
        )

        if len(newROI_proc) > 0:
            NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(
                right_poly_line,
                newROI_proc,
                minX,
                maxY
            )
        else:
            break

        if isEmpty != 1:
            if IsPleuraLikeComponent(NextComponent, poly_line=poly_line2):
                cv2.drawContours(interpreted_Image, [NextComponent], 0, (255, 0, 0), 1)
                right_contour_components.append(NextComponent)

                last_Y = Ymax
                Ymax = int(max(PY_))
                minX = int(min(PX_) - 10)
                minX = max(0, minX)

                newROI = orig_Image[minX:maxX, Ymax:img.shape[1] - 1]
            else:
                print("[RIGHT] Componenta respinsa.")
                break
        else:
            break

    # -------------------------
    # RIGHT EXTENSION
    # -------------------------

    base_components_for_extension = [PrincipalComponent] + left_contour_components + right_contour_components

    right_extension = ExtendRightPleuraByPrediction(
        orig_Image,
        poly_line2,
        base_components_for_extension
    )

    if right_extension is not None:
        print(f"[RIGHT_EXTENSION] Extensie-contur adaugata cu {len(right_extension)} puncte.")
        right_contour_components.append(right_extension)
        cv2.drawContours(interpreted_Image, [right_extension], -1, (255, 255, 0), 1)
    else:
        print("[RIGHT_EXTENSION] Nu a fost nevoie sau nu s-a gasit extensie valida.")

    raw_components = [PrincipalComponent] + right_contour_components + left_contour_components

    continuous_contour = BuildContinuousContourFromComponents(
        raw_components,
        image_shape=orig_Image.shape
    )

    if continuous_contour is not None:
        print("[CONTINUOUS] Contur continuu final construit.")
        cv2.drawContours(interpreted_Image, [continuous_contour], -1, (0, 255, 255), 1)
        final_components = [continuous_contour]
        ctr = continuous_contour
    else:
        print("[CONTINUOUS] Nu s-a putut construi contur continuu. Folosesc componente brute.")
        final_components = raw_components
        ctr = PrincipalComponent

    final_mask = np.zeros(interpreted_Image.shape[:2], dtype=np.uint8)
    cv2.drawContours(final_mask, [ctr], -1, 255, -1)

    return ctr, final_components, raw_components, orig_Image, final_mask


def ExtractPleuralLineAuto(crop_Image_rgb):
    global CONTINUOUS_MAX_GAP_PX
    global CONTINUOUS_MAX_CENTER_DIFF_PX
    global RIGHT_EXTENSION_ENABLED

    presets = [
        {
            "name": "default",
            "percentile": 88,
            "top_ratio": 0.08,
            "bottom_ratio": 0.65,
            "gap": 120,
            "center_diff": 45,
            "right_extension": True,
            "clahe_clip": 2.0,
            "kernel_h_width": 21,
            "kernel_v_height": 25
        },
        {
            "name": "sensitive_low_threshold",
            "percentile": 82,
            "top_ratio": 0.05,
            "bottom_ratio": 0.70,
            "gap": 140,
            "center_diff": 55,
            "right_extension": True,
            "clahe_clip": 2.2,
            "kernel_h_width": 25,
            "kernel_v_height": 25
        },
        {
            "name": "very_sensitive",
            "percentile": 78,
            "top_ratio": 0.03,
            "bottom_ratio": 0.72,
            "gap": 160,
            "center_diff": 65,
            "right_extension": True,
            "clahe_clip": 2.5,
            "kernel_h_width": 29,
            "kernel_v_height": 25
        },
        {
            "name": "strict_clean",
            "percentile": 92,
            "top_ratio": 0.08,
            "bottom_ratio": 0.60,
            "gap": 90,
            "center_diff": 35,
            "right_extension": False,
            "clahe_clip": 1.8,
            "kernel_h_width": 19,
            "kernel_v_height": 30
        },
        {
            "name": "wider_roi",
            "percentile": 85,
            "top_ratio": 0.00,
            "bottom_ratio": 0.80,
            "gap": 150,
            "center_diff": 60,
            "right_extension": True,
            "clahe_clip": 2.2,
            "kernel_h_width": 25,
            "kernel_v_height": 25
        },
        {
            "name": "upper_focus",
            "percentile": 84,
            "top_ratio": 0.00,
            "bottom_ratio": 0.55,
            "gap": 130,
            "center_diff": 50,
            "right_extension": True,
            "clahe_clip": 2.3,
            "kernel_h_width": 23,
            "kernel_v_height": 25
        }
    ]

    best = None
    best_score = -1e9
    best_preset_name = None

    original_gap = CONTINUOUS_MAX_GAP_PX
    original_center_diff = CONTINUOUS_MAX_CENTER_DIFF_PX
    original_right_extension = RIGHT_EXTENSION_ENABLED

    try:
        for preset in presets:
            try:
                CONTINUOUS_MAX_GAP_PX = preset["gap"]
                CONTINUOUS_MAX_CENTER_DIFF_PX = preset["center_diff"]
                RIGHT_EXTENSION_ENABLED = preset["right_extension"]

                interpreted_Image = crop_Image_rgb.copy()

                ctr, final_components, raw_components, img, final_mask = ExtractPleuralLine(
                    crop_Image_rgb,
                    interpreted_Image,
                    prep_percentile=preset["percentile"],
                    prep_top_ratio=preset["top_ratio"],
                    prep_bottom_ratio=preset["bottom_ratio"],
                    prep_clahe_clip=preset["clahe_clip"],
                    prep_kernel_h_width=preset["kernel_h_width"],
                    prep_kernel_v_height=preset["kernel_v_height"]
                )

                score = ScoreContourQuality(ctr, crop_Image_rgb.shape)

                print(
                    f"  [AUTO] preset={preset['name']}, "
                    f"score={score:.2f}, "
                    f"components={len(final_components)}"
                )

                if score > best_score:
                    best_score = score
                    best_preset_name = preset["name"]
                    best = (
                        ctr,
                        final_components,
                        raw_components,
                        img,
                        final_mask,
                        interpreted_Image
                    )

            except Exception as e:
                print(f"  [AUTO] preset={preset['name']} esuat: {e}")
                continue

    finally:
        CONTINUOUS_MAX_GAP_PX = original_gap
        CONTINUOUS_MAX_CENTER_DIFF_PX = original_center_diff
        RIGHT_EXTENSION_ENABLED = original_right_extension

    if best is None:
        raise ValueError("[ExtractPleuralLineAuto] Niciun preset nu a reusit.")

    print(f"  [AUTO] Best preset: {best_preset_name}, score={best_score:.2f}")

    ctr, final_components, raw_components, img, final_mask, interpreted_Image = best

    return (
        ctr,
        final_components,
        raw_components,
        img,
        final_mask,
        interpreted_Image,
        best_preset_name,
        best_score
    )


# ============================================================
# WIDTH AND IRREGULARITY
# ============================================================

def GetVerticals(x, y):
    xall = range(np.min(x), np.max(x))

    verticals = np.zeros(len(xall), dtype=int)
    vert_mean = np.zeros(len(xall), dtype=int)

    x_plot = range(len(xall) + np.min(x))
    offset = len(x_plot[np.min(x):])

    ii = -1

    for iv in xall:
        yindex = np.argwhere(x == iv)
        Y_point = y[yindex]

        ii += 1

        if len(Y_point) == 0:
            continue

        y_first = int(Y_point.flat[0])
        y_last = int(Y_point.flat[-1])

        verticals[ii] = np.abs(y_first - y_last)
        vert_mean[ii] = np.abs(y_first + y_last) / 2

    return verticals, vert_mean, offset


def PeakValleyDiff(x, y, verticals, vert_mean):
    xall = range(np.min(x), np.max(x))

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

    peaks, _ = find_peaks(vert_mean)

    figure(num=None, figsize=(18, 2))
    plt.plot(vert_mean, 'forestgreen')
    plt.plot(peaks, vert_mean[peaks], "r^")

    valleys, _ = find_peaks(-vert_mean)

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

    if np.mean(Dif) == 0:
        PVDif = 0
    else:
        PVDif = 1 / np.mean(Dif) * 100

    return PVDif, offset


def Width_and_Irreg(orig_Image, ctr):
    x = []
    y = []

    for i in ctr:
        for j in i:
            x.append(j[0])
            y.append(j[1])

    x = np.array(x)
    y = np.array(np.max(y) - y)

    verticals, vert_mean, offset = GetVerticals(x, y)

    WidthPleuraPixels = np.mean(verticals)
    PleuraVariation = np.var(verticals)

    PVDif, offset = PeakValleyDiff(x, y, verticals, vert_mean)

    print("\n[Width_and_Irreg]")
    print("Pleural width in pixels:", WidthPleuraPixels)
    print("Pleural width variance:", PleuraVariation)
    print("Peaks-Valleys difference:", PVDif)

    return np.max(y) - vert_mean, offset


# ============================================================
# INTERRUPTIONS
# ============================================================

def Interruptions(imgo, middle_line, offset, components):
    print("[Interruptions] Temporar pastrat simplificat. Conturul final continuu este in output.")


# ============================================================
# MAIN BATCH
# ============================================================

def main():
    input_dir = r'C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES'
    output_root = r'C:\Facultate\AN4\Licenta\Licenta-Cod\DEBUG_OUT_BATCH'

    os.makedirs(output_root, exist_ok=True)

    filled_all_dir = os.path.join(output_root, "ALL_07_FILLED_CONTOUR_OVERLAYS")
    os.makedirs(filled_all_dir, exist_ok=True)

    start_idx = 0
    end_idx = 61

    success_count = 0
    fail_count = 0
    failed_images = []

    print("\n" + "=" * 70)
    print("BATCH TEST PLEURA - AUTO PRESETS")
    print(f"Input : {input_dir}")
    print(f"Output: {output_root}")
    print(f"Folder comun 07: {filled_all_dir}")
    print("=" * 70 + "\n")

    for idx in range(start_idx, end_idx + 1):
        img_name = f"{idx}.jpg"
        img_path = os.path.join(input_dir, img_name)

        image_output_dir = os.path.join(output_root, f"{idx:02d}")
        os.makedirs(image_output_dir, exist_ok=True)

        print(f"\n[{idx}/{end_idx}] Procesez: {img_name}")

        if not os.path.exists(img_path):
            print(f"  [SKIP] Nu exista fisierul: {img_path}")
            fail_count += 1
            failed_images.append((img_name, "Fisier inexistent"))
            continue

        try:
            # -----------------------------
            # 1. Citire imagine
            # -----------------------------
            orig_Image = io.imread(img_path)

            if orig_Image.ndim == 2:
                orig_Image = cv2.cvtColor(orig_Image, cv2.COLOR_GRAY2RGB)
            elif orig_Image.ndim == 3 and orig_Image.shape[2] == 4:
                orig_Image = cv2.cvtColor(orig_Image, cv2.COLOR_RGBA2RGB)

            cv2.imwrite(
                os.path.join(image_output_dir, '01_original.png'),
                cv2.cvtColor(orig_Image, cv2.COLOR_RGB2BGR)
            )

            # -----------------------------
            # 2. Pixel converter
            # -----------------------------
            one_pixel = PixelConverter(orig_Image)

            if one_pixel <= 0 or one_pixel > 0.5:
                print("  [WARN] PixelConverter suspect. Folosesc fallback.")
                one_pixel = ONE_PIXEL_FALLBACK

            print(f"  one_pixel = {one_pixel:.4f} mm")

            # -----------------------------
            # 3. Crop
            # -----------------------------
            crop_Image = CropBorder(orig_Image)

            cv2.imwrite(
                os.path.join(image_output_dir, '02_cropped.png'),
                crop_Image
            )

            crop_Image_rgb = cv2.cvtColor(crop_Image, cv2.COLOR_GRAY2RGB)

            # -----------------------------
            # 4. Detectie pleura automata cu preseturi multiple
            # -----------------------------
            (
                ctr,
                final_components,
                raw_components,
                img,
                final_mask,
                interpreted_Image,
                best_preset_name,
                best_score
            ) = ExtractPleuralLineAuto(crop_Image_rgb)

            # -----------------------------
            # 5. Salvare debug componente colorate
            # -----------------------------
            cv2.imwrite(
                os.path.join(image_output_dir, '03_components_colored.png'),
                cv2.cvtColor(interpreted_Image, cv2.COLOR_RGB2BGR)
            )

            cv2.imwrite(
                os.path.join(image_output_dir, '03b_final_continuous_mask_FILLED.png'),
                final_mask
            )

            # -----------------------------
            # 6. Overlay componente brute
            # -----------------------------
            raw_overlay_open = draw_components_as_open_lines(
                crop_Image_rgb,
                raw_components,
                color=(255, 0, 0),
                thickness=1
            )

            cv2.imwrite(
                os.path.join(image_output_dir, '04a_raw_components_OPEN.png'),
                cv2.cvtColor(raw_overlay_open, cv2.COLOR_RGB2BGR)
            )

            # -----------------------------
            # 7. Overlay contur final continuu
            # -----------------------------
            final_overlay_open = draw_components_as_open_lines(
                crop_Image_rgb,
                final_components,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.imwrite(
                os.path.join(image_output_dir, '04_final_overlay_CONTINUOUS.png'),
                cv2.cvtColor(final_overlay_open, cv2.COLOR_RGB2BGR)
            )

            # -----------------------------
            # 8. Masca contur deschis final
            # -----------------------------
            direct_mask_open = build_open_line_mask(
                crop_Image_rgb.shape[:2],
                final_components,
                thickness=2
            )

            cv2.imwrite(
                os.path.join(image_output_dir, '05_direct_components_mask_CONTINUOUS.png'),
                direct_mask_open
            )

            # -----------------------------
            # 9. Overlay verde final
            # -----------------------------
            green_overlay_open = draw_components_as_open_lines(
                crop_Image_rgb,
                final_components,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.imwrite(
                os.path.join(image_output_dir, '06_green_overlay_CONTINUOUS.png'),
                cv2.cvtColor(green_overlay_open, cv2.COLOR_RGB2BGR)
            )

            # -----------------------------
            # 10. Contur final peste imagine
            # -----------------------------
            filled_overlay = crop_Image_rgb.copy()

            cv2.drawContours(
                filled_overlay,
                [ctr],
                -1,
                (0, 255, 0),
                1
            )

            filled_overlay_path = os.path.join(
                image_output_dir,
                '07_filled_contour_overlay.png'
            )

            cv2.imwrite(
                filled_overlay_path,
                cv2.cvtColor(filled_overlay, cv2.COLOR_RGB2BGR)
            )

            filled_overlay_all_path = os.path.join(
                filled_all_dir,
                f'{idx:02d}_filled_contour_overlay.png'
            )

            cv2.imwrite(
                filled_overlay_all_path,
                cv2.cvtColor(filled_overlay, cv2.COLOR_RGB2BGR)
            )

            # -----------------------------
            # 11. Info numeric
            # -----------------------------
            info_path = os.path.join(image_output_dir, 'info.txt')

            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Imagine: {img_name}\n")
                f.write(f"one_pixel: {one_pixel:.6f} mm/pixel\n")
                f.write(f"crop_shape: {crop_Image.shape}\n")
                f.write(f"best_preset: {best_preset_name}\n")
                f.write(f"best_score: {best_score:.4f}\n")
                f.write(f"raw_components: {len(raw_components)}\n")
                f.write(f"final_components: {len(final_components)}\n")
                f.write(f"07_filled_individual: {filled_overlay_path}\n")
                f.write(f"07_filled_folder_comun: {filled_overlay_all_path}\n\n")

                for c_idx, comp in enumerate(raw_components):
                    area = cv2.contourArea(comp)
                    f.write(
                        f"RAW component {c_idx}: "
                        f"points={len(comp)}, area={area:.2f}\n"
                    )

                for c_idx, comp in enumerate(final_components):
                    area = cv2.contourArea(comp)
                    f.write(
                        f"FINAL component {c_idx}: "
                        f"points={len(comp)}, area={area:.2f}\n"
                    )

            success_count += 1

            print(f"  [OK] Salvat in: {image_output_dir}")
            print(f"  [OK] 07 copiat in folderul comun: {filled_overlay_all_path}")

        except Exception as e:
            fail_count += 1
            failed_images.append((img_name, str(e)))

            print(f"  [ESEC] {img_name}: {e}")

            error_path = os.path.join(image_output_dir, 'error.txt')

            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Eroare la imaginea: {img_name}\n")
                f.write(str(e))
                f.write("\n\n")
                f.write(traceback.format_exc())

            continue

    # -----------------------------
    # REZUMAT FINAL
    # -----------------------------
    summary_path = os.path.join(output_root, 'summary.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BATCH TEST PLEURA - AUTO PRESETS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Interval imagini: {start_idx}-{end_idx}\n")
        f.write(f"Reusite: {success_count}\n")
        f.write(f"Esuate: {fail_count}\n")
        f.write("Folder comun 07_filled_contour_overlay:\n")
        f.write(f"{filled_all_dir}\n\n")

        if failed_images:
            f.write("Imagini esuate:\n")
            for name, err in failed_images:
                f.write(f"  {name}: {err}\n")

    print("\n" + "=" * 70)
    print("BATCH TERMINAT")
    print(f"Reusite: {success_count}")
    print(f"Esuate : {fail_count}")
    print(f"Rezumat salvat in: {summary_path}")
    print(f"Toate imaginile 07 sunt salvate in: {filled_all_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
