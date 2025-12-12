import cv2
import numpy as np
mas = cv2. imread('1mac.jpg', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mas, 220, 255, cv2.THRESH_BINARY)
kernal = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(mask, kernal, iterations=2)
mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
cv2.imshow('output', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
