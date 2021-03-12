import cv2 as cv
import numpy as np

# rescale function
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# read image
img = cv.imread('media/kor.jpg')
img_resized = rescaleFrame(img, scale=.3)

# grayscale
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# blur image
blur = cv.GaussianBlur(img_resized, (5,5), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)

# edge cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow('canny', canny)

# dilating image
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('dilated', dilated)

# eroding
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('erode', eroded)

# resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('resized', resized)

#cropped
cropped = resized[50:200, 200:400]
cv.imshow('cropped', cropped)

# show image
cv.imshow('color', img_resized)

cv.waitKey(0)