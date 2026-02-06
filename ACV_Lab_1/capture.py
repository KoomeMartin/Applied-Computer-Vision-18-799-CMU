import cv2
import os
 
""" 
Collect images for camera calibration using a chessboard pattern
This script captures images from the webcam and saves them for calibration, but you can use this with any camera.
"""
 
SAVE_DIR = "calib_images" # Directory to save images
NUM_IMAGES = 10  # Number of images, use at least 15-20 images for better calibration
CHESSBOARD_SIZE = (9, 7) # How many inner corners per chessboard row and column, these are not the squares
CAMERA_INDEX = 0  # Default camera index (0 for built-in webcam)
 
os.makedirs(SAVE_DIR, exist_ok=True)

# Print controls and configuration info
print("=" * 50)
print("CAMERA CALIBRATION - Image Capture Tool")
print("=" * 50)
print(f"\nConfiguration:")
print(f"  - Save directory: {SAVE_DIR}")
print(f"  - Images to capture: {NUM_IMAGES}")
print(f"  - Chessboard size: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
print(f"\nControls:")
print(f"  [SPACE] - Capture image (when chessboard is detected)")
print(f"  [ESC]   - Exit program")
print(f"\nTips:")
print(f"  - Hold the chessboard steady when green corners appear")
print(f"  - Vary the angle and distance for each capture")
print(f"  - Ensure good lighting for better detection")
print("=" * 50)
print("\nStarting camera... Press ESC to quit.\n")

# create video capture object
cap = cv2.VideoCapture(CAMERA_INDEX)

# checking if the camera successfully opened
if not cap.isOpened():
    print("ERROR: Could not open camera. Check your webcam connection.")
    exit(1)

count = 0 # counter to store the number of images captured
 
while True:
    ret, frame = cap.read() # read a frame from the camera and ret is for return value, while the frame is the image. The return value is true if the frame is read correctly
    if not ret:
        break # if frame is not read correctly, break the loop
 
    display = frame.copy() # copy the frame to display
    
    # This section of code does image processing to find chessboard corners
    # First convert the frame to grayscale for better boxes detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    #  Find chessboard corners with findChessboardCorners function
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
 
    # If any corners are found, draw them 
    if found:
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        cv2.putText(display, "Press SPACE to save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # create the window and show the frame
    cv2.imshow("Webcam", display)
    key = cv2.waitKey(1) # wait for a key press for 1 ms this helps to refresh the frame
 
    # If SPACE is pressed and corners are found, save the image
    if key == 32 and found:  
        filename = os.path.join(SAVE_DIR, f"img_{count:02d}.jpg")
        cv2.imwrite(filename, frame) # using the imwrite function to save the image
        print(f"Saved {filename}")
        count += 1
        # If image number reaches the limit, stop the process
        if count >= NUM_IMAGES:
            print(f"\n✓ Captured all {NUM_IMAGES} images. Exiting...")
            break
 
    # If ESC is pressed, exit the loop
    elif key == 27:
        print(f"\nExiting... Captured {count}/{NUM_IMAGES} images.")
        break
 
cap.release()
cv2.destroyAllWindows()
print("Camera released. Done!")