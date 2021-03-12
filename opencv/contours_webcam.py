import cv2 as cv
import numpy as np

# rescale function
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture(1)

while True:
    isTrue, frame = capture.read()
    cv.imshow('webcam', frame)

    blank = np.zeros(frame.shape, dtype='uint8')

    # gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)

    # blur
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    cv.imshow('blur', blur)

    # canny 
    canny = cv.Canny(blur, 125, 175)
    cv.imshow('canny', canny)

    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(blank, contours, -1, (0,0,255), 1)
    cv.imshow('contours drawn', blank)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()