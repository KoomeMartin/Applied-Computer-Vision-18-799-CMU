import cv2
img = cv2.imread('image.jpg')
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.circle(img, (100, 100), 40, (255, 0, 0), -1)
cv2.imshow('Drawn Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
