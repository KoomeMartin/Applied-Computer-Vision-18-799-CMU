import cv2
img = cv2.imread('image.jpg')
print('Image shape:', img.shape)
print('Pixel value at (100, 100):', img[100, 100])
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
