import numpy as np
import cv2
#Import necessary functions only

def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points
	# Use Direct Linear Transform (DLT)
	
	N = x1.shape[0]
	
	# Build the A matrix
	A = np.zeros((2*N, 9))
	
	for i in range(N):
		x, y = x2[i]
		u, v = x1[i]
		
		A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
		A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
	
	# Compute SVD
	_, _, Vt = np.linalg.svd(A)
	V = Vt.T
	
	# H is the last column of V
	H2to1 = V[:, -1].reshape(3, 3)
	
	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	centroid1 = np.mean(x1, axis=0)
	centroid2 = np.mean(x2, axis=0)
	
	#Shift the origin of the points to the centroid
	x1_shifted = x1 - centroid1
	x2_shifted = x2 - centroid2
	
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_dist1 = np.max(np.linalg.norm(x1_shifted, axis=1))
	max_dist2 = np.max(np.linalg.norm(x2_shifted, axis=1))
	
	scale1 = np.sqrt(2) / max_dist1
	scale2 = np.sqrt(2) / max_dist2
	
	x1_norm = x1_shifted * scale1
	x2_norm = x2_shifted * scale2
	
	#Similarity transform 1
	T1 = np.array([
		[scale1, 0, -scale1 * centroid1[0]],
		[0, scale1, -scale1 * centroid1[1]],
		[0, 0, 1]
	])
	
	#Similarity transform 2
	T2 = np.array([
		[scale2, 0, -scale2 * centroid2[0]],
		[0, scale2, -scale2 * centroid2[1]],
		[0, 0, 1]
	])
	
	#Compute homography on normalized coordinates
	H_norm = computeH(x1_norm, x2_norm)
	
	#Denormalization: H = T1^(-1) * H_norm * T2
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2
	
	return H2to1




def computeH_ransac(x1, x2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	
	N = x1.shape[0]
	num_iterations = 2000  # Increased iterations for better convergence
	threshold = 5.0  # threshold for inlier detection 
	best_inlier_count = 0
	bestH2to1 = None
	best_inliers_mask = None
	
	for iteration in range(num_iterations):
		# Randomly select 4 point pairs (minimum required for homography)
		if N < 4:
			break
		
		random_indices = np.random.choice(N, 4, replace=False)
		x1_sample = x1[random_indices]
		x2_sample = x2[random_indices]
		
		# Compute homography from sample
		try:
			H = computeH_norm(x1_sample, x2_sample)
		except:
			continue
		
		# Convert x2 to homogeneous coordinates and apply H
		x2_homo = np.hstack([x2, np.ones((N, 1))])
		x1_proj_homo = (H @ x2_homo.T).T
		
		# Convert back from homogeneous coordinates
		x1_proj = x1_proj_homo[:, :2] / x1_proj_homo[:, 2:3]
		
		# Compute reprojection error
		errors = np.linalg.norm(x1 - x1_proj, axis=1)
		
		# Find inliers (points with error below threshold)
		inliers_mask = errors < threshold
		inlier_count = np.sum(inliers_mask)
		
		# Update best homography if this one has more inliers
		if inlier_count > best_inlier_count:
			best_inlier_count = inlier_count
			bestH2to1 = H
			best_inliers_mask = inliers_mask
	
	# Refine homography with all inliers if we have enough
	if best_inliers_mask is not None and np.sum(best_inliers_mask) >= 4:
		x1_inliers = x1[best_inliers_mask]
		x2_inliers = x2[best_inliers_mask]
		bestH2to1 = computeH_norm(x1_inliers, x2_inliers)
	
	# Convert inliers mask to 0/1 vector
	inliers = best_inliers_mask.astype(int) if best_inliers_mask is not None else np.zeros(N, dtype=int)
	
	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	
	# Invert the homography (from template to image coordinates)
	H_inv = np.linalg.inv(H2to1)
	
	# Get dimensions of the image
	h_img, w_img = img.shape[:2]
	h_template, w_template = template.shape[:2]
	
	#Create mask of same size as template (white = 255, black = 0)
	mask = np.ones((h_template, w_template), dtype=np.uint8) * 255
	
	#Warp mask by appropriate homography (inverse to go from image to template space)
	warped_mask = cv2.warpPerspective(mask, H_inv, (w_img, h_img))
	
	#Warp template by appropriate homography
	if len(template.shape) == 3:
		warped_template = cv2.warpPerspective(template, H_inv, (w_img, h_img))
	else:
		warped_template = cv2.warpPerspective(template, H_inv, (w_img, h_img))
	
	#Use mask to combine the warped template and the image
	# Convert mask to 3 channels if image is color
	if len(img.shape) == 3:
		mask_3ch = np.stack([warped_mask, warped_mask, warped_mask], axis=2)
		# Normalize mask to 0-1
		mask_normalized = mask_3ch.astype(np.float32) / 255.0
		# Blend: where mask is 1, use warped_template; where mask is 0, use img
		composite_img = (warped_template.astype(np.float32) * mask_normalized + 
						img.astype(np.float32) * (1 - mask_normalized)).astype(np.uint8)
	else:
		mask_normalized = warped_mask.astype(np.float32) / 255.0
		composite_img = (warped_template.astype(np.float32) * mask_normalized + 
						img.astype(np.float32) * (1 - mask_normalized)).astype(np.uint8)
	
	return composite_img


