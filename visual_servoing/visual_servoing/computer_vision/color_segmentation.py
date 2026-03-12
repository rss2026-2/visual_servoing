import cv2
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

testing = False

def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def erosion_filter(box_size = 3, iterations = 1):
    erosion_kernel = np.ones((box_size, box_size), np.uint8)
    def erosion_func(input_image):
        return cv2.erode(input_image, erosion_kernel, iterations = iterations)
    
    return erosion_func

def dilation_filter(box_size = 3, iterations = 1):
    dilation_kernel = np.ones((box_size, box_size), np.uint8)
    def dilation_func(input_image):
        return cv2.dilate(input_image, dilation_kernel, iterations = iterations)
    
    return dilation_func

def create_filter_cascade(filter_list):
    def filter_cascade(image):
        for filt in filter_list:
            image = filt(image)

        return image
    

    return filter_cascade

def filter_list_from_filter_specs(filter_specs):
    filter_lis = []
    switch, sizes, iterations = filter_specs["switch"], filter_specs["sizes"], filter_specs["iterations"]
    for i,val in enumerate(switch):
        if val == 1:
            if i % 2 == 0: # erosion filter
                filter_lis.append(erosion_filter(box_size = sizes[i], iterations = iterations[i]))
            else: # dilation filter
                filter_lis.append(dilation_filter(box_size = sizes[i], iterations = iterations[i]))
    
    return filter_lis


def cd_color_segmentation(img, template, distances = None, filter_specs = None):
    """
    Implement the cone detection using color segmentation algorithm
    Input:
        img: np.3darray; the input image with a cone to be detected. BGR.
        template: Not required, but can optionally be used to automate setting hue filter values.
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
            (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    """
    ########## YOUR CODE STARTS HERE ##########
    if distances is None:
        distances = [0.40395939, 0.21454798, 0.60708237] #tuned parameters

    if filter_specs is None:
        filter_specs = {
            "switch": [1,1,0,0,0,0],
            "sizes": [5,5,0,0,0,0],
            "iterations": [1,2,0,0,0,0]
        }

    avg_template_hsv = get_hsv_from_template(template)
    hsv_range = get_hsv_range_by_distance(avg_template_hsv, distances)

    hsv_input_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_input_img, hsv_range["lower"], hsv_range["upper"])

    filter_cascade = create_filter_cascade(
        filter_list_from_filter_specs(filter_specs)
    )

    filtered_mask = filter_cascade(color_mask)
 
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    box_max = 0
    bounds_max = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if testing:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
        if w * h > box_max:
            bounds_max = ((x,y), (x+w, y+h))
            box_max = w * h

    if testing:
        cv2.rectangle(img, bounds_max[0], bounds_max[1], (0, 0, 255), 2)
        image_print(img)

    bounding_box = bounds_max

    ########### YOUR CODE ENDS HERE ########### 
    # Return bounding box
    return bounding_box


def get_cone_image(img_num = None):
    """
    Input:
        img_num: The number corresponding to the cone img

    Returns:
        The image corresponding to the given img number or a random cone image if no number is given
    """
    test_imgs_dir = "./visual_servoing/computer_vision/test_images_cone"
    if img_num is None:
        img_num = np.random.randint(1, 21)

    img_file = test_imgs_dir + "/test" + str(img_num) + ".jpg"
    return cv2.imread(img_file)

def get_hsv_from_template(template_img):
    """
    Finds the average hsv of the template image

    Input:
        template_img: The template image that the color segmentation is looking for

    Returns:
        The average hsv color of the template_img
    """
    hsv_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2HSV)

    white_mask = (hsv_img[:,:,0] == 0) & (hsv_img[:,:,1] == 0) & (hsv_img[:,:,2] == 255)
    opaque_img = hsv_img[~white_mask]

    avg_hsv = opaque_img.mean(axis=0)

    return avg_hsv

def get_hsv_range_by_distance(hsv, distances):
    hue_dist_below, hue_dist_above = distances[0]
    sat_dist_below, sat_dist_above = distances[1]
    val_dist_below, val_dist_above = distances[2]

    hue_max, sat_max, val_max = 179, 255, 255

    hue, sat, val = hsv
    
    lower_bound = np.array([
        hue - (hue * hue_dist_below),
        sat - (sat * sat_dist_below),
        val - (val * val_dist_below)
    ])

    upper_bound = np.array([
        hue + (hue_max - hue) * hue_dist_above,
        sat + (sat_max - sat) * sat_dist_above,
        val + (val_max - val) * val_dist_above
    ])
    
    return {"lower": lower_bound, "upper": upper_bound}

# def get_hsv_range_by_distance(hsv, distances):
#     """
#     Constructs a range of hsv colors based on the distances of the hue, saturation, and value from an average hsv value

#     Input:
#         hsv: The average hsv value,
#         distances: A 1x3 array with distances in between 0 and 1 for the hue, saturation, and value
    
#     Returns:
#         A dictionary with keys {"lower", "upper"} corresponding to the lowest and highest hsv value within the range
#     """
#     hue_dist, sat_dist, val_dist = distances
#     hue_max = 179
#     sat_max = 255
#     val_max = 255

#     hue, sat, val = hsv
#     lower_bound = np.array([
#         max(0, hue - hue_dist*hue_max), 
#         max(0, sat - sat_dist*sat_max), 
#         max(0, val - val_dist*val_max)
#     ])
#     upper_bound = np.array([
#         min(hue_max, hue + hue_dist*hue_max), 
#         min(sat_max, sat + sat_dist*sat_max), 
#         min(val_max, val + val_dist*val_max)
#     ])

#     return {"lower": lower_bound, "upper": upper_bound}

if testing:
    cone_image = get_cone_image(17)
    test_range = [0.40395939, 0.21454798, 0.60708237]
    cd_color_segmentation(cone_image, cv2.imread("./visual_servoing/computer_vision/test_images_cone/cone_template.png"), distances = test_range)