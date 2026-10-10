import cv2
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray, 100, 200)
cv2.imshow('Threshold', thresh)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
