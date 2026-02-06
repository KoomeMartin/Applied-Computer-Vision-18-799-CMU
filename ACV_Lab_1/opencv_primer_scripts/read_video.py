import cv2
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
