#import lxr as love

import numpy as np
import cv2
from skimage import morphology
import matplotlib.pyplot as plt


def load_image(image_number):
    Img = cv2.imread('nighttime.png') # nighttime lane line
    return Img

def gradient_thresh(img, image_number):  # test_realworld picture:150, 180 gazebo:230, 255 rosbag: 200,255
    """
    Apply sobel edge detection on input image in x, y direction
    """
    # 1. Convert the image to gray scale
    # 2. Gaussian blur the image
    # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    # 4. Use cv2.addWeighted() to combine the results
    # 5. Convert each pixel to unint8, then apply threshold to get binary image

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 1


    thresh_min = 200
    thresh_max = 255

    gray_blurred_img = cv2.GaussianBlur(gray_img, ksize=(5, 5), sigmaX=0)  # 2

    grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F, 1, 0, ksize=5)  # 3
    grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, .5, 0)  # 4
    img_uint8 = grad
    binary_output = np.zeros(img_uint8.shape)  # 5
    binary_output = ((img_uint8[:, :] >= thresh_min).astype(np.uint8) & (img_uint8[:, :] <= thresh_max).astype(np.uint8))

    # for column in range(img_uint8.shape[0]):
    # 	for row in range(img_uint8.shape[1]):
    # 		# print(img_uint8[column][row])
    # 		binary_output[column][row] = 1 if img_uint8[column][row]>=thresh_min and img_uint8[column][row]<=thresh_max else 0

    # binary_output_255 = 255 * binary_output
    # cv2.imwrite('1_gardient_thresh.png',binary_output_255)
    return binary_output

def color_thresh(img, image_number):  # test real world picture: 100,255 gazebo: 130,160 rosbag: 210, 250
    """thresh=(100, 255)
    Convert RGB to HSL and threshold to binary image using S channel
    """
    # 1. Convert the image from RGB to HSL
    img_HSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2. Apply threshold on S channel to get binary image
    if image_number == 1:
        thresh = (100, 255)
    elif image_number == 2:
        thresh = (130, 160)
    elif image_number == 3:
        thresh = (210, 250)
    img_h = img_HSL.shape[0]
    img_w = img_HSL.shape[1]
    binary_output = np.zeros([img_h, img_w])
    binary_output = (img_HSL[:, :, 1] > thresh[0]).astype(np.uint8)
    # print(binary_output)

    # for i in range(img_h):
    # 	for j in range(img_w):
    # 		if img_HSL[i][j][1] < thresh[0]:
    # 			binary_output[i][j] = 0
    # 		else:
    # 			binary_output[i][j] = 1

    # binary_output_255 = 255 * binary_output
    # cv2.imwrite('1_color_thresh.png',binary_output_255)
    return binary_output

def combinedBinaryImage(img, image_number):
    """
    Get combined binary image from color filter and sobel filter
    """
    # 1. Apply sobel filter and color filter on input image
    # 2. Combine the outputs
    ## Here you can use as many methods as you want.
    ColorOutput = color_thresh(img, image_number)
    SobelOutput = gradient_thresh(img, image_number)
    binaryImage = np.zeros(SobelOutput.shape)
    binaryImage[(ColorOutput == 1) & (SobelOutput == 1)] = 1
    # Remove noise from binary image
    binaryImage = (morphology.remove_small_objects(binaryImage.astype('bool'), min_size=50, connectivity=2)).astype(
        np.uint8)
    return binaryImage, SobelOutput, ColorOutput

def visualize_img(Img,img_name):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(img_name)
    # ax.imshow(np.array(Img / 255).astype(float))
    ax.imshow(Img)
    plt.show()

def perspective_transform(img, image_number):
    """
    Get bird's eye view from input image
    """
    # 1. Visually determine 4 source points and 4 destination points
    # All points are in format [cols, rows]
    # pt_A = [187, 144]
    # pt_B = [81, 210]
    # pt_C = [393, 210]
    # pt_D = [236, 144]

    pt_A = [492, 350]  # zuoshang
    pt_B = [384, 398]  # zuoxia
    pt_C = [939, 398]  # youxia
    pt_D = [884, 350]  # youshang


    # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # 3. Generate warped image in bird view using cv2.warpPerspective()
    out_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # out_img_255 = 255 * out_img
    # cv2.imwrite('1_out_img.png',out_img_255)
    # cv2.imshow('output_img',out_img)
    # cv2.waitKey(0)

    return out_img, np.linalg.inv(M)

def line_fit(binary_warped):
    """
    Find and fit lane lines
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image

    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype('uint8')
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[0:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # leftx_base = np.argmax(histogram[:midpoint]) + 25
    # rightx_base = np.argmax(histogram[midpoint:]) + 25
    # print(midpoint, binary_warped.shape)
    # Choose the number of sliding windows
    nwindows = 9

    window_noise_threshold = 1

    # Set height of windows
    window_height = int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = np.zeros((nwindows, 2), dtype=int)
    right_lane_inds = np.zeros((nwindows, 2), dtype=int)
    top_left_wind_left = np.zeros((nwindows, 2), dtype=int)
    bottom_right_wind_left = np.zeros((nwindows, 2), dtype=int)
    top_left_wind_right = np.zeros((nwindows, 2), dtype=int)
    bottom_right_wind_right = np.zeros((nwindows, 2), dtype=int)

    width = binary_warped.shape[0]
    height = binary_warped.shape[1]

    window_picture_right = np.zeros((margin, 11))
    window_picture_right_test = np.zeros((2 * margin, 20))

    # np.set_printoptions(threshold=sys.maxsize)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        ##TO DO
        # if leftx_current - margin/2 < 0: # exceeds the left side
        # 	top_left_wind[window] = (0, (window+1)*window_height)
        # 	bottom_right_wind[window] = (margin, window*window_height)
        # elif leftx_current + margin/2 >= binary_warped.shape[0]: # exceeds the right side
        # 	top_left_wind[window] = (binary_warped.shape[0] - margin, (window+1)*window_height)
        # 	bottom_right_wind[window] = (binary_warped.shape[0],  window*window_height)
        # else: #Regular case
        # 	top_left_wind[window] = (leftx_current - margin/2, (window+1)*window_height)
        # 	bottom_right_wind[window] = (leftx_current + margin/2, window*window_height)

        # print(leftx_base, rightx_base, leftx_current, rightx_current)

        if leftx_current - margin / 2 < 0:  # exceeds the left side
            top_left_wind_left[window][0] = 0
            top_left_wind_left[window][1] = window * window_height
            bottom_right_wind_left[window][0] = margin
            bottom_right_wind_left[window][1] = (window + 1) * window_height
        # print("left exceeds the left side")
        elif leftx_current + margin / 2 >= binary_warped.shape[1]:  # exceeds the right side
            top_left_wind_left[window][0] = binary_warped.shape[1] - margin
            top_left_wind_left[window][1] = window * window_height
            bottom_right_wind_left[window][0] = binary_warped.shape[1]
            bottom_right_wind_left[window][1] = (window + 1) * window_height
        # print("left exceeds the right side")
        else:  # Regular case
            top_left_wind_left[window][0] = leftx_current - margin / 2
            top_left_wind_left[window][1] = window * window_height
            bottom_right_wind_left[window][0] = leftx_current + margin / 2
            bottom_right_wind_left[window][1] = (window + 1) * window_height
        # print("left exceeds the regular side")

        if rightx_current - margin / 2 < 0:  # exceeds the left side
            top_left_wind_right[window][0] = 0
            top_left_wind_right[window][1] = window * window_height
            bottom_right_wind_right[window][0] = margin
            bottom_right_wind_right[window][1] = (window + 1) * window_height
        # print("right exceeds the left side")
        elif (rightx_current + margin / 2 >= binary_warped.shape[1]):  # exceeds the right side
            top_left_wind_right[window][0] = (binary_warped.shape[1] - margin)
            top_left_wind_right[window][1] = window * window_height
            bottom_right_wind_right[window][0] = (binary_warped.shape[1])
            bottom_right_wind_right[window][1] = ((window + 1) * window_height)
        # print("right exceeds the right side")
        else:  # Regular case
            top_left_wind_right[window][0] = (rightx_current - margin / 2)
            top_left_wind_right[window][1] = (window * window_height)
            bottom_right_wind_right[window][0] = (rightx_current + margin / 2)
            bottom_right_wind_right[window][1] = ((window + 1) * window_height)
        # print("right exceeds the regular side")

        ####
        # Draw the windows on the visualization image using cv2.rectangle()
        ##TO DO
        color = (255, 255, 255)
        thickness = 2
        out_img = cv2.rectangle(binary_warped, (top_left_wind_left[window][0], top_left_wind_left[window][1]),
                                (bottom_right_wind_left[window][0], bottom_right_wind_left[window][1]), color,
                                thickness)
        out_img = cv2.rectangle(out_img, (top_left_wind_right[window][0], top_left_wind_right[window][1]),
                                (bottom_right_wind_right[window][0], bottom_right_wind_right[window][1]), color,
                                thickness)
        # Displaying the image
        # cv2.imshow('output_img2', out_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ####
        # Identify the nonzero pixels in x and y within the window
        ##TO DO
        # not done yet, need to get the x location of the lane pixels in the window
        # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # midpoint = np.int(histogram.shape[0]/2)
        # leftx_base = np.argmax(histogram[100:midpoint]) + 100
        # rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

        # print(binary_warped.shape)
        # print(top_left_wind_right[window][0], bottom_right_wind_right[window][0], top_left_wind_right[window][1], bottom_right_wind_right[window][1])
        window_picture_right = binary_warped[top_left_wind_right[window][1]: bottom_right_wind_right[window][1],
                               top_left_wind_right[window][0]:bottom_right_wind_right[window][0]]
        # print(window_picture_right)
        window_nonzero_right = window_picture_right.nonzero()
        # print(window_nonzero_right, window_noise_threshold)
        if (window_nonzero_right[0].size >= window_noise_threshold):
            window_nonzeroy_right = np.array(window_nonzero_right[1])
            window_nonzerox_right = np.array(window_nonzero_right[0])
            right_lane_inds[window][0] = np.mean(window_nonzeroy_right) + top_left_wind_right[window][0]
            right_lane_inds[window][1] = np.mean(window_nonzerox_right) + top_left_wind_right[window][1]
            rightx_current = right_lane_inds[window][0]
        # print(">thres: " , right_lane_inds)
        else:
            if (window == 1):
                right_lane_inds[window][1] = right_lane_inds[window - 1][1]
                right_lane_inds[window][0] = right_lane_inds[window - 1][0]
                rightx_current = right_lane_inds[window][0]
            # print("<thres (win1): " , right_lane_inds)
            elif (window == 0):
                right_lane_inds[window][0] = rightx_base
                right_lane_inds[window][1] = window_height / 2
                rightx_current = right_lane_inds[window][0]
            # print("<thres (win0): " , right_lane_inds)
            else:
                # right_lane_inds[window][1] = right_lane_inds[window - 1][1]
                # right_lane_inds[window][0] = right_lane_inds[window - 1][0]
                right_lane_inds[window][1] = 2 * right_lane_inds[window - 1][1] - right_lane_inds[window - 2][1]
                right_lane_inds[window][0] = 2 * right_lane_inds[window - 1][0] - right_lane_inds[window - 2][0]
                rightx_current = right_lane_inds[window][0]
            # print("<thres (win>2): " , right_lane_inds)

        window_picture_left = binary_warped[top_left_wind_left[window][1]: bottom_right_wind_left[window][1],
                              top_left_wind_left[window][0]:bottom_right_wind_left[window][0]]
        window_nonzero_left = window_picture_left.nonzero()
        if (window_nonzero_left[0].size >= window_noise_threshold):
            window_nonzeroy_left = np.array(window_nonzero_left[1])
            window_nonzerox_left = np.array(window_nonzero_left[0])
            left_lane_inds[window][0] = np.mean(window_nonzeroy_left) + top_left_wind_left[window][0]
            left_lane_inds[window][1] = np.mean(window_nonzerox_left) + top_left_wind_left[window][1]
            leftx_current = left_lane_inds[window][0]
        else:
            if (window == 1):
                left_lane_inds[window][1] = left_lane_inds[window - 1][1]
                left_lane_inds[window][0] = left_lane_inds[window - 1][0]
                leftx_current = left_lane_inds[window][0]
            elif (window == 0):
                left_lane_inds[window][0] = leftx_base
                left_lane_inds[window][1] = window_height / 2
                leftx_current = left_lane_inds[window][0]
            else:
                # left_lane_inds[window][1] = left_lane_inds[window - 1][1]
                # left_lane_inds[window][0] = left_lane_inds[window - 1][0]
                left_lane_inds[window][1] = 2 * left_lane_inds[window - 1][1] - left_lane_inds[window - 2][1]
                left_lane_inds[window][0] = 2 * left_lane_inds[window - 1][0] - left_lane_inds[window - 2][0]
                leftx_current = left_lane_inds[window][0]

        ####
        # Append these indices to the lists
        ##TO DO

        ####
        # If you found > minpix pixels, recenter next window on their mean position
        ##TO DO
        leftx_current = left_lane_inds[window][0]
        rightx_current = right_lane_inds[window][0]
    ####

    # # Concatenate the arrays of indices
    # left_lane_inds = np.concatenate(left_lane_inds)
    # right_lane_inds = np.concatenate(right_lane_inds)

    # # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using np.polyfit()
    # If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
    # the second order polynomial is unable to be sovled.
    # Thus, it is unable to detect edges.
    # print(right_lane_inds.shape)
    # print(right_lane_inds[:, 0])

    try:
        ##TODO
        left_fit = np.polyfit(left_lane_inds[:, 1], left_lane_inds[:, 0], deg=2)
        right_fit = np.polyfit(right_lane_inds[:, 1], right_lane_inds[:, 0], deg=2)
        # print(left_fit)
        # print(right_fit)
    ####
    except TypeError:
        print("Unable to detect lanes")
        return None

    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret

def bird_fit(binary_warped, ret):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	return result

def final_viz(undist, ret, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result

if __name__ == '__main__':

    image_number = 1


    img = load_image(image_number)
    visualize_img(img,'img')
    binaryImage, SobelOutput, ColorOutput = combinedBinaryImage(img, image_number)
    visualize_img(ColorOutput, 'ColorOutput')
    visualize_img(SobelOutput, 'SobelOutput')
    visualize_img(binaryImage, 'binaryImage')
    img_birdeye, Minv = perspective_transform(binaryImage, image_number)
    visualize_img(img_birdeye,'img_birdeye')
    ret = line_fit(img_birdeye)
    visualize_img(ret['out_img'],'lane line detection')
    bird_fit_img = bird_fit(img_birdeye, ret)
    visualize_img(bird_fit_img, 'bird_fit_img')
    combine_fit_img = final_viz(img, ret, Minv)
    visualize_img(combine_fit_img, 'combine_fit_img')
    img = cv2.imread('night.jpeg')
    combine_fit_img = final_viz(img, ret, Minv)
    visualize_img(combine_fit_img, 'combine_fit_img_for_night')