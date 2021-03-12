import cv2 as cv
import numpy as np

# rescale function
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# read image
img = cv.imread('media/cat.jpg')
img_resized = rescaleFrame(img, scale=.3)

# draw on image
cv.rectangle(img_resized, (10,10), (50, 50), (0,255,0), thickness=2)
cv.line(img_resized, (20,60), (70,100), (255,255,255), thickness=2)
# text
cv.putText(img_resized, 'catty', (60,30), cv.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 1)

# show image
cv.imshow('cat, lines, text', img_resized)

cv.waitKey(0)

# read video or webcam

#capture = cv.VideoCapture(1)

#while True:
#    isTrue, frame = capture.read()
#    cv.imshow('webcam', frame)

#    if cv.waitKey(20) & 0xFF==ord('d'):
#        break

#capture.release()
#cv.destroyAllWindows()

