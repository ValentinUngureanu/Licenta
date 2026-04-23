import math
import re

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
from skimage.util import img_as_ubyte
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen


### Crop image border:

def CropBorder(orig_Image):
    orig_img = orig_Image.copy()

    gray_Image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    (thresh, img_bin) = cv2.threshold(gray_Image, 128, 255, cv2.THRESH_BINARY | cv2.ADAPTIVE_THRESH_MEAN_C)

    # Converting image to a binary image
    _, threshold = cv2.threshold(gray_Image, 110, 255,
                                 cv2.THRESH_BINARY)

    # Detecting shapes in image by selecting region with same colors or intensity
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Searching through every region selected to find the required polygon
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area
        if area > 1000:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

            # Checking if the number of sides of the selected region is 4
            if (len(approx) == 4):
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    # Converting image to a binary image
    _, threshold = cv2.threshold(gray_Image, 110, 255,
                                 cv2.THRESH_BINARY)

    # Detecting shapes in image by selecting region with same colors or intensity
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    black = np.zeros_like(img_bin)

    # Searching through every region selected to find the required polygon
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area
        if area > 1000:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

            # Checking if the no. of sides of the selected region is 4
            if (len(approx) == 4):
                cv2.drawContours(black, [approx], 0, 255, -1)

            # Checking if the number of sides of the selected region is 2
            if (len(approx) == 2):
                cv2.drawContours(black, [approx], 0, 255, -1)

    # Get left boundary coordinate
    white_pixels = np.array(np.where(black == 255))
    last_small = white_pixels[1, white_pixels[1] < 100]

    if len(white_pixels[1]) == 0 or len(last_small) == 0:
        left_bound = 25

    if len(white_pixels[1]) > 0 and len(last_small) != 0:
        left_bound = last_small[-1]

    # Right and top-bottom boundary coordinates

    # Searching through every region selected to find the required polygon
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area
        if area < 30:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)

            # Checking if the number of sides of the selected region is 4
            if (len(approx) == 4):
                cv2.drawContours(orig_img, [approx], 0, 255, -1)

    # Converting image to a binary image
    _, threshold = cv2.threshold(gray_Image, 110, 255,
                                 cv2.THRESH_BINARY)

    # Detecting shapes in image by selecting region with same colors or intensity
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    black = np.zeros_like(img_bin)

    # Searching through every region selected to find the required polygon
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area
        if area < 30:
            approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)

            # Checking if the no. of sides of the selected region is 4
            if (len(approx) == 4):
                cv2.drawContours(black, [approx], 0, 255, -1)

            # Checking if the number of sides of the selected region is 2
            if (len(approx) == 2):
                cv2.drawContours(black, [approx], 0, 255, -1)

    img = black

    # Defining a kernel length
    kernel_length = 2

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(black, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Get left and top-bottom boundary coordinates

    bar_image = horizontal_lines_img

    columns = np.zeros(bar_image.shape[1], dtype=int)

    for i in range(bar_image.shape[1]):
        columns[i] = np.count_nonzero(bar_image[:, i])

    bar_pos = np.where(columns == np.max(columns))
    bar_pos = bar_pos[0][0]

    # bar column as array
    bar = bar_image[:, bar_pos] // 255

    bar_pixels = np.array(np.where(bar == 1))
    first_bar_pixel = bar_pixels[:, 0]
    last_bar_pixel = bar_pixels[:, -1]

    # Crop the image

    if len(white_pixels[1]) == 0 or len(last_small) == 0:
        print(
            "Warning: Nonconforming image, intensity bar not found, image croped more on the right and slightly on the left")
        crop_img = gray_Image[first_bar_pixel[0]:last_bar_pixel[0], left_bound:bar_pos - 20].copy()

    if first_bar_pixel[0] == 0 or last_bar_pixel[0] == 0:
        print("Error: Nonconforming image, image not cropped")
        crop_img = gray_Image.copy()

    if len(white_pixels[1]) > 0 and len(last_small) != 0:
        crop_img = gray_Image[first_bar_pixel[0]:last_bar_pixel[0], left_bound:bar_pos - 20].copy()

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

    tes.pytesseract.tesseract_cmd =  r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
    p30 = np.poly1d(np.polyfit(Y_, X_, order))
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
    matrx = np.zeros(shape=(len(points) - 1, len(points) - 1))
    for i in range(0, len(points) - 1):
        for j in range(0, len(points) - 1):
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


def Binarize(Image, tresh):
    if len(Image.shape) == 3:
        Image = rgb2gray(Image)
    binarized = Image < tresh
    return binarized


def PreprocessImage(Image):
    Image = rgb2gray(Image)
    return Image


def CropImage(Image):
    yellow_pix_x, yellow_pix_y = GetYellowPix(Image)
    if len(yellow_pix_x) > 0:
        min_x = min(yellow_pix_x)
        max_y = max(yellow_pix_y)
        return Image[min_x: 600, 0:max_y - 30], min_x
    else:
        return Image, 0


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
    graph = csr_matrix(distances)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    max_nr = 0
    lbl = 0

    for i in range(0, n_components - 1):
        nr_pts = np.count_nonzero(labels == i)
        if nr_pts > max_nr:
            max_nr = nr_pts
            lbl = i

    for i in range(0, len(points) - 1):
        if labels[i] == lbl:
            X_.append(points[i].x)
            Y_.append(points[i].y)
            pts.append(points[i])
    return pts, X_, Y_


def ConnectedContour(ROI):
    img = img_as_ubyte(ROI)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img) > 0:
        thresh = threshold_yen(img) - 0.2 * threshold_yen(img)

        binary = img > thresh
        img = binary.astype(np.uint8)  # convert to an unsigned byte
        img *= 255

        # img = cv2.erode(img,np.ones((3,3),np.uint8),iterations = 1)

        new_img = np.zeros_like(img)

        for val in np.unique(img)[1:]:
            mask = np.uint8(img == val)
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            new_img[labels == largest_label] = val

        img = cv2.dilate(new_img, np.ones((3, 3), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return contours
    else:
        return False


def IdentifyPrincipalContour(img):
    points = list()

    ExtractContour(img, points)

    distances = removeOutliers(points, 50)

    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)
    poly_line, PX_ = IdnetifyPoly(PX_, PY_, 3)

    deviation = 20
    pleural_underline = Fit(pts, poly_line, deviation)

    for p in pleural_underline:
        img[p.x + 10:img.shape[0] - 1, p.y:p.y + 1] = 0

    Xmin = min(p.x for p in pleural_underline) - 10
    Ymin = min(p.y for p in pleural_underline) - 10
    Xmax = max(p.x for p in pleural_underline) + 10
    Ymax = max(p.y for p in pleural_underline) + 10

    ROI = img[Xmin:Xmax, Ymin:Ymax]

    contours = ConnectedContour(ROI)
    PX = []
    PY = []
    pts = []

    class Point:
        x = 0.0
        y = 0.0

    for k in contours:
        for i in k:
            for j in i:
                p = Point()
                p.x = j[1] + Xmin
                p.y = j[0] + Ymin
                pts.append(p)
                PX.append(p.x)
                PY.append(p.y)

    deviation = 50
    pleura, PX_, PY_ = Fit2(pts, poly_line, deviation, PX, PY)

    poly_line2, PX_ = IdnetifyPoly(PX_, PY_, 1)

    ps = []
    for p in pleura:
        ps.append(tuple([p.y, p.x]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)
    return ctr, PY_, poly_line2, pleura


def IdentifySecondaryContour(lateral_poly, img, minX, minY):
    points = list()

    ExtractContour(img, points)

    distances = removeOutliers(points, 50)

    pts, PX_, PY_ = ExtractConnectedComponents(distances, points)

    if len(PX_) < 1:
        return False, False, False, False, 1

    poly_line, PX_ = IdnetifyPoly(PX_, PY_, 3)

    deviation = 30
    pleural_underline = Fit(pts, poly_line, deviation)

    for p in pleural_underline:
        img[p.x + 10:img.shape[0] - 1, p.y:p.y + 1] = 0

    Xmin = min(p.x for p in pleural_underline) - 20
    Ymin = min(p.y for p in pleural_underline) - 10
    Xmax = max(p.x for p in pleural_underline) + 10
    Ymax = max(p.y for p in pleural_underline) + 10

    if Xmin < 0: Xmin = 0
    if Ymin < 0: Ymin = 0
    if Xmax > img.shape[0] - 1: Xmax = img.shape[0] - 1
    if Ymax > img.shape[1] - 1: Ymax = img.shape[1] - 1
    ROI = img[Xmin:Xmax, Ymin:Ymax]

    contours = ConnectedContour(ROI)

    if contours != False:
        PX = []
        PY = []
        pts = []

        class Point:
            x = 0.0
            y = 0.0

        for k in contours:
            for i in k:
                for j in i:
                    p = Point()
                    p.x = j[1] + Xmin + minX
                    p.y = j[0] + Ymin + minY
                    pts.append(p)
                    PX.append(p.x + Xmin + minX)
                    PY.append(p.y + Ymin + minY)

        deviation = 100
        pleura, PX_, PY_ = Fit2(pts, lateral_poly, deviation, PX_, PY_)

        ps = []
        for p in pleura:
            ps.append(tuple([p.y, p.x]))

        if len(pleura) > 0:
            ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)
            return ctr, PY_, PX_, pleura, 0
        else:
            return False, False, False, False, 1
    else:
        return False, False, False, False, 1


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

    im = Image.fromarray(img)
    im = im.convert('P', palette=Image.ADAPTIVE, colors=7)
    img = np.array(im)
    # normalizeaza intre 0-255 inainte de rgb2gray
    img = (img / img.max() * 255).astype(np.uint8)
    img = np.stack([img, img, img], axis=-1)
    img = rgb2gray(img)
    img = Binarize(img, 0.4)


    left_contour_components = []
    right_contour_components = []
    contour_components = []
    contours = []

    PrincipalComponent, PYS, poly_line2, points = IdentifyPrincipalContour(img)

    contours += points
    cv2.drawContours(interpreted_Image, [PrincipalComponent], 0, (0, 255, 0), 1)

    Ymin = min(PYS) + 10
    Ex = []
    Ey = []
    for y in range(0, Ymin):
        Ey.append(y)
        x_hat = poly_line2(y)
        Ex.append(x_hat)

    minX = int(min(Ex)) - 40
    maxX = img.shape[0] - 1
    left_poly_line, Ex = IdnetifyPoly(Ex, Ey, 3)

    newROI = orig_Image[minX: maxX, 0:Ymin]

    last_Y = Ymin + 5
    while Ymin < last_Y:

        im = Image.fromarray(newROI)
        im = im.convert('P', palette=Image.ADAPTIVE, colors=10)
        newROI = np.array(im)
        newROI = np.stack([newROI, newROI, newROI], axis=-1)

        newROI = rgb2gray(newROI)

        newROI = Binarize(newROI, 0.4)
        if len(newROI) > 0:
            NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(left_poly_line, newROI, minX, 0)
        else:
            Ymin += 1000

        if isEmpty != 1:
            contours += points
            cv2.drawContours(interpreted_Image, [NextComponent], 0, (50, 150, 255), 1)
            left_contour_components.append(NextComponent)
            last_Y = Ymin
            Ymin = int(min(PY_) + 10)
            minX = int(min(PX_) - 10)
            newROI = orig_Image[minX: maxX, 0:Ymin]

        else:
            Ymin += 1000

    Ymax = max(PYS) - 10
    Ex = []
    Ey = []
    for y in range(Ymax, img.shape[1] - 1):
        Ey.append(y)
        x_hat = poly_line2(y)
        Ex.append(x_hat)

    minX = int(min(Ex)) - 30
    maxX = img.shape[0] - 1
    maxY = Ymax
    right_poly_line, Ex = IdnetifyPoly(Ex, Ey, 3)
    newROI = orig_Image[minX: maxX, Ymax: img.shape[1] - 1]

    last_Y = Ymax - 5
    while Ymax > last_Y:

        im = Image.fromarray(newROI)
        im = im.convert('P', palette=Image.ADAPTIVE, colors=10)

        newROI = np.array(im)
        newROI = np.stack([newROI, newROI, newROI], axis=-1)

        newROI = rgb2gray(newROI)

        newROI = Binarize(newROI, 0.4)

        if len(newROI) > 0:
            NextComponent, PY_, PX_, points, isEmpty = IdentifySecondaryContour(right_poly_line, newROI, minX, maxY)
        else:
            Ymax -= 1000

        if isEmpty != 1:
            contours += points
            cv2.drawContours(interpreted_Image, [NextComponent], 0, (255, 0, 0), 1)
            right_contour_components.append(NextComponent)
            last_Y = Ymax
            Ymax = int(max(PY_))
            minX = int(min(PX_) - 10)
            newROI = orig_Image[minX: maxX, Ymax:img.shape[1] - 1]
        else:
            Ymax -= 1000

    mask = np.zeros([interpreted_Image.shape[0], interpreted_Image.shape[1]], np.uint8)

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
            p1, p2, p3, p4, p5, p6 = MergeLeftComponent(right_contour_components[i + 1], right_contour_components[i])
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

    cnt = ConnectedContour(mask)
    if cnt != False:
        ps = []
        for k in cnt:
            for i in k:
                for j in i:
                    ps.append(tuple([j[0], j[1] + MINx]))

    ctr = np.array(ps).reshape((-1, 1, 2)).astype(np.int32)

    components = []
    components.append(PrincipalComponent)

    for comp in range(len(right_contour_components)):
        components.append(right_contour_components[comp])

    for comp in range(len(left_contour_components)):
        components.append(left_contour_components[comp])

    return ctr, components, orig_Image


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
        verticals[ii] = np.abs(int(Y_point.flat[0]) - int(Y_point.flat[-1]))
        vert_mean[ii] = np.abs(int(Y_point.flat[0]) + int(Y_point.flat[-1])) / 2

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

# ### MAIN:
def main():
    orig_Image = io.imread('./ORIGINAL_IMAGES/1.jpg')

    new_image=CropBorder(orig_Image)
    plt.imshow(new_image)
    plt.show()



if __name__ == '__main__':
    main()


# orig_Image = io.imread('./Images/IMG-0021-00001.jpg')

# # How many mm is one pixel
# one_pixel = PixelConverter(orig_Image)
# if one_pixel!=1:
#     print('One pixel is:', one_pixel, 'mm')

# # Crop function
# crop_Image = CropBorder(orig_Image)

# # Compare the two images
# figure(num=None, figsize=(20, 20))
# plt.imshow(orig_Image, cmap = 'gray')

# figure(num=None, figsize=(20, 20))
# plt.imshow(crop_Image, cmap = 'gray')


# interpreted_Image = orig_Image.copy() # not sure if interpreted image is needed

# ctr, components, img = ExtractPleuralLine(orig_Image,interpreted_Image)

# middle_line, offset = Width_and_Irreg(img, ctr)

# Interruptions(img, middle_line, offset, components)

# #fig, ax1 = plt.subplots(1, 1, figsize = (15, 5), dpi = 150)
# #cv2.drawContours(orig_Image,[ctr],-1,(255,255,255),2)
# #ax1.imshow(img)