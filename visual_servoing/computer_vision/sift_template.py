import cv2
import imutils
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################


def image_print(img):
    """
    Helper function to print out images, for debugging.
    Press any key to continue.
    """
    winname = "Image"
    cv2.namedWindow(winname)         # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cd_sift_ransac(img, template):
    """
    Implement the cone detection using SIFT + RANSAC algorithm.
    Input:
        img: np.3darray; the input image with a cone to be detected
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box in image coordinates (Y increasing downwards),
            where (x1, y1) is the top-left pixel of the box
            and (x2, y2) is the bottom-right pixel of the box.
    """
    # Minimum number of matching features
    MIN_MATCH = 10  # Adjust this value as needed
    # Create SIFT
    sift = cv2.SIFT_create()

    # Compute SIFT on template and test image
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img, None)

    # Find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Find and store good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance: # Lowe's ratio test
            # accept match m if it is singinifactly better than the second best match
            good.append(m)

    # If enough good matches, find bounding box
    if len(good) > MIN_MATCH:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Create mask
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = template.shape[0], template.shape[1]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # use homography M to map pts onto img
        corner_pts = cv2.perspectiveTransform(pts, M)

        # find min/max across corner points
        x_min, x_max = corner_pts[...,0].min(), corner_pts[...,0].max()
        y_min, y_max = corner_pts[...,1].min(), corner_pts[...,1].max()

        h0, w0, _ = img.shape
        # clamp to original image
        x_min = max(0,x_min)
        x_max = min(w0,x_max)
        y_min = max(0,y_min)
        y_max = min(h0,y_max)

        # Return bounding box
        return ((x_min, y_min), (x_max, y_max))
    else:
        print(f"[SIFT] not enough matches; matches: {len(good)}")

        # Return bounding box of area 0 if no match found
        return ((0, 0), (0, 0))


def cd_template_matching(img, template):
    """
    Implement the cone detection using template matching algorithm.
    Input:
        img: np.3darray; the input image with a cone to be detected
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box in px (Y increases downward),
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    """
    template_canny = cv2.Canny(template, 50, 200)

    # Perform Canny Edge detection on test image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(grey_img, 50, 200)

    # Get dimensions of template
    (img_height, img_width) = img_canny.shape[:2]

    # Keep track of best-fit match
    best_match = (-float("inf"),(0,0),(0,0))

    # Loop over different scales of image
    for scale in np.linspace(1.5, .5, 50):
        # Resize the image
        resized_template = imutils.resize(
            template_canny, width=int(template_canny.shape[1] * scale))
        (h, w) = resized_template.shape[:2]
        # Check to see if test image is now smaller than template image
        if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
            continue

        # Use OpenCV template matching functions to find the best match
        # across template scales.

        method = cv2.TM_CCOEFF_NORMED # returns values between 0 and 1
        res = cv2.matchTemplate(img_canny,resized_template,method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_match[0]:
            best_match = (max_val, max_loc, (w,h))

    # Remember to resize the bounding box using the highest scoring scale
    # x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
    bounding_box = (best_match[1],
                    (best_match[1][0]+best_match[2][0],best_match[1][1]+best_match[2][1])
                    )
    return bounding_box

def show_image(img, bounding_box, alg):
    new_img = img.copy()
    pt1 = (int(bounding_box[0][0]), int(bounding_box[0][1]))
    pt2 = (int(bounding_box[1][0]), int(bounding_box[1][1]))
    cv2.rectangle(new_img, pt1, pt2, (255, 0, 0), 2)
    cv2.putText(new_img, alg, (pt1[0], pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(f"{alg}.png", new_img)

citgo_template = cv2.imread("./test_images_citgo/citgo_template.png",0)
img = cv2.imread("./test_images_citgo/citgo1.jpeg")
bb = cd_sift_ransac(img, citgo_template)
show_image(img, bb, "sift")

bb = cd_template_matching(img, citgo_template)
show_image(img, bb, "template")
