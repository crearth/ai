import cv2 as cv
import numpy as np

# rescale function
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# read and scale image
img = rescaleFrame(cv.imread('media/kor.jpg'), scale=.3)

blank = np.zeros(img.shape, dtype='uint8')

# gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', gray)

# blur
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)

# canny 
canny = cv.Canny(blur, 125, 175)
cv.imshow('canny', canny)

# threshold
#ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
#cv.imshow('thresh', thresh)

# coordinates of contours in image, hierarchie of contours = (, mode of returning contours (retr=all, rert_external=only outside, rert_tree=all hierarchical contours), )
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('contours drawn', blank)

# show image
cv.imshow('kor', img)
cv.waitKey(0)