import cv2 as c
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image = c.imread('3.jpg')
imgs = c.resize(image, (332, 250))
print(imgs.shape)
img = c.cvtColor(imgs, c.COLOR_BGR2RGB)
# 1.converting colour image into grayscale image
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)
cl = c.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
CLAHE = cl.apply(gray)
# 2.optic detection and removal
_, mask = c.threshold(gray, 145, 225, c.THRESH_BINARY)
kernal = np.ones((5, 5), np.uint8)
OD = c.dilate(mask, kernal, iterations=2)
ODdetec = c.bitwise_not(OD)
ODremove = c.bitwise_and(gray, ODdetec)
# 3.blood vessel and fovea segmentation
b, green_fundus, r = c.split(img)
clahe = c.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_enhanced_green_fundus = clahe.apply(green_fundus)
# applying alternate sequential filtering (3 times closing opening)
r1 = c.morphologyEx(contrast_enhanced_green_fundus, c.MORPH_OPEN, c.getStructuringElement(c.MORPH_ELLIPSE, (5, 5)),
                    iterations=1)
R1 = c.morphologyEx(r1, c.MORPH_CLOSE, c.getStructuringElement(c.MORPH_ELLIPSE, (5, 5)), iterations=1)
r2 = c.morphologyEx(R1, c.MORPH_OPEN, c.getStructuringElement(c.MORPH_ELLIPSE, (11, 11)), iterations=1)
R2 = c.morphologyEx(r2, c.MORPH_CLOSE, c.getStructuringElement(c.MORPH_ELLIPSE, (11, 11)), iterations=1)
r3 = c.morphologyEx(R2, c.MORPH_OPEN, c.getStructuringElement(c.MORPH_ELLIPSE, (23, 23)), iterations=1)
R3 = c.morphologyEx(r3, c.MORPH_CLOSE, c.getStructuringElement(c.MORPH_ELLIPSE, (23, 23)), iterations=1)
f4 = c.subtract(R3, contrast_enhanced_green_fundus)
f5 = clahe.apply(f4)
# removing very small contours through area parameter noise removal
ret, f6 = c.threshold(f5, 15, 255, c.THRESH_BINARY)
mask = np.ones(f5.shape[:2], dtype="uint8") * 255
contours, hierarchy = c.findContours(f6, c.RETR_LIST, c.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if c.contourArea(cnt) <= 200:
        c.drawContours(mask, [cnt], -1, 0, -1)
im = c.bitwise_and(f5, f5, mask=mask)
ret, fin = c.threshold(im, 15, 255, c.THRESH_BINARY_INV)
newfin = c.erode(fin, c.getStructuringElement(c.MORPH_ELLIPSE, (3, 3)), iterations=1)
# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
# vessels and also in an interval of area
fundus_eroded = c.bitwise_not(newfin)
xmask = np.ones(img.shape[:2], dtype="uint8") * 255
xcontours, xhierarchy = c.findContours(fundus_eroded, c.RETR_LIST, c.CHAIN_APPROX_SIMPLE)
for cnt in xcontours:
    shape = "unidentified"
    peri = c.arcLength(cnt, True)
    approx = c.approxPolyDP(cnt, 0.04 * peri, False)
    if len(approx) > 4 and 3000 >= c.contourArea(cnt) >= 100:
        shape = "circle"
    else:
        shape = "veins"
    if shape == "circle":
        c.drawContours(xmask, [cnt], -1, 0, -1)
finimage = c.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
BVdetec = c.bitwise_not(finimage)

# 4.Edge detection - blood vessel and fovea removal
BVremove = c.bitwise_and(gray, BVdetec)
ODBV = c.bitwise_and(ODremove, BVdetec)
# 5.
blur = c.GaussianBlur(CLAHE, (5,5), 0)
# Apply Sobelx in high output datatype 'float32'
# and then converting back to 8-bit to prevent overflow
sobelx_64 = c.Sobel(blur, c.CV_32F, 1, 0, ksize=3)
absx_64 = np.absolute(sobelx_64)
sobelx_8u1 = absx_64 / absx_64.max() * 255
sobelx_8u = np.uint8(sobelx_8u1)

# Similarly for Sobely
sobely_64 = c.Sobel(blur, c.CV_32F, 0, 1, ksize=3)
absy_64 = np.absolute(sobely_64)
sobely_8u1 = absy_64 / absy_64.max() * 255
sobely_8u = np.uint8(sobely_8u1)

# From gradients calculate the magnitude and changing
# it to 8-bit (Optional)
mag = np.hypot(sobelx_8u, sobely_8u)
mag = mag / mag.max() * 255
mag = np.uint8(mag)

# Find the direction and change it to degree
theta = np.arctan2(sobely_64, sobelx_64)
angle = np.rad2deg(theta)

# Find the neighbouring pixels (b,c) in the rounded gradient direction
# and then apply non-max suppression
M, N = mag.shape
Non_max = np.zeros((M, N), dtype=np.uint8)

for i in range(1, M - 1):
    for j in range(1, N - 1):
        # Horizontal 0
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180) or (-22.5 <= angle[i, j] < 0) or (
                -180 <= angle[i, j] < -157.5):
            b = mag[i, j + 1]
            c = mag[i, j - 1]
        # Diagonal 45
        elif (22.5 <= angle[i, j] < 67.5) or (-157.5 <= angle[i, j] < -112.5):
            b = mag[i + 1, j + 1]
            c = mag[i - 1, j - 1]
        # Vertical 90
        elif (67.5 <= angle[i, j] < 112.5) or (-112.5 <= angle[i, j] < -67.5):
            b = mag[i + 1, j]
            c = mag[i - 1, j]
        # Diagonal 135
        elif (112.5 <= angle[i, j] < 157.5) or (-67.5 <= angle[i, j] < -22.5):
            b = mag[i + 1, j - 1]
            c = mag[i - 1, j + 1]

            # Non-max Suppression
        if (mag[i, j] >= b) and (mag[i, j] >= c):
            Non_max[i, j] = mag[i, j]
        else:
            Non_max[i, j] = 0

# Set high and low threshold
highThreshold = 21
lowThreshold = 15

M, N = Non_max.shape
out = np.zeros((M, N), dtype=np.uint8)
# If edge intensity is greater than 'High' it is a sure-edge
# below 'low' threshold, it is a sure non-edge
strong_i, strong_j = np.where(Non_max >= highThreshold)
zeros_i, zeros_j = np.where(Non_max < lowThreshold)

# weak edges
weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

# Set same intensity value for all edge pixels
out[strong_i, strong_j] = 255
out[zeros_i, zeros_j] = 0
out[weak_i, weak_j] = 75

M, N = out.shape
for i in range(1, M - 1):
    for j in range(1, N - 1):
        if out[i, j] == 75:
            if 255 in [out[i + 1, j - 1], out[i + 1, j], out[i + 1, j + 1], out[i, j - 1], out[i, j + 1],
                       out[i - 1, j - 1], out[i - 1, j], out[i - 1, j + 1]]:
                out[i, j] = 255
            else:
                out[i, j] = 0

cany = Image.fromarray(out)
cany.save('3c.jpg')

titles = ['image', 'gray', 'ODdetec', 'ODremove', 'BVdetec', 'BVremove', 'ODBV', 'CLAHE']
images = [img, gray, ODdetec, ODremove, BVdetec, BVremove, ODBV, CLAHE]

for i in range(8):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
