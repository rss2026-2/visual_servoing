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


def cd_color_segmentation(img, template = None, hsv_range = None, filter_specs = None, detection_mode = "cone", margins = None):
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

    ### Trial Data ###
    """
        Ranges:
            Hue: [0.22555346595529566, 0.37443759630528706]
            Saturation: [0.09102723799465517, 1.0]
            Value: [0.48860718137548237, 1.0]
        Avg: 0.8722486753384867
        Min: 0.8250783699059561
        Target: 0.6314762637129322
        Filter Specs: {'switch': array([0, 1, 0, 0, 0, 0]), 'sizes': array([7, 5, 7, 2, 4, 7]), 'iterations': array([1, 2, 2, 3, 2, 1])}
    """

    ### Tuned Parameters ###
    if hsv_range is None:
        if detection_mode == "line":
            hsv_range = get_hsv_range_by_colors(
                hsv_low = hsv_convert_to_cv2((20, 20, 55)),
                hsv_high =  hsv_convert_to_cv2((29, 90, 100))
            )
        elif detection_mode == "cone":
            hsv_range = {
                'lower': (np.float64(0.0), np.float64(231.6560756120999), np.float64(122.47303168670507)), 
                'upper': (np.float64(74.62069078649506), np.float64(255.0), np.float64(255.0))
                }
    
    if filter_specs is None:
        filter_specs = {
                "switch": [0, 1], # erosion: off, dilation: on
                "sizes" : [0, 5], # erosion: box_size 0, dilation: box_size 4
                "iterations": [0, 2] # erosion: iterations 0, dilation: iterations 1
        }
    if margins is None:
        if detection_mode == "cone":
            margins = (0, 3)
        elif detection_mode == "line":
            margins = (100, 200) #(x_margin, y_margin)
    
    ### Program ###
    hsv_input_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_input_img, hsv_range["lower"], hsv_range["upper"])

    cv2.imwrite("visual_servoing/computer_vision/output_images/color_mask.png", color_mask)

    filter_cascade = create_filter_cascade(
        filter_list_from_filter_specs(filter_specs)
    )

    filtered_mask = filter_cascade(color_mask)

    cv2.imwrite("visual_servoing/computer_vision/output_images/filtered_mask.png", filtered_mask)
 
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    init_ctr_image = img.copy()
    for ctr in contours:
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(init_ctr_image, (x,y), (x+w,y+h), (255,0,0), 2)
    
    biggest_ctr =  max(contours, key = cv2.contourArea)
    bx,by,bw,bh = cv2.boundingRect(biggest_ctr)

    cv2.rectangle(init_ctr_image, (bx,by), (bx+bw,by+bh), (0,0,255), 2)
    cv2.imwrite("visual_servoing/computer_vision/output_images/initial_contours.png", init_ctr_image)


    x_margin = margins[0]
    y_margin = margins[1]

    img_height, img_width, _= img.shape
    bounding_box = None
    if len(contours) != 0:
        biggest_ctr = max(contours, key = cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(biggest_ctr)

        left, right = max(bx - x_margin, 0), min(bx + bw + x_margin, img_width-1)
        top, bottom = max(by - y_margin, 0), min(by+bh+y_margin, img_height-1)

        cv2.rectangle(init_ctr_image, (left, top), (right,bottom), (0,255,0), 2)
        cv2.imwrite("visual_servoing/computer_vision/output_images/region_of_interest.png", init_ctr_image)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        region_of_interest_mask = filtered_mask[top:bottom, left:right].copy()
        
        region_of_interest_mask = cv2.morphologyEx(region_of_interest_mask, cv2.MORPH_CLOSE, kernel)

        contours_in_roi, _ = cv2.findContours(region_of_interest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for ctr in contours_in_roi:
            for point in ctr:
                point[0] = (point[0][0] + left, point[0][1] + top)
        
        new_ctr_image = img.copy()
        for ctr in contours_in_roi:
            x,y,w,h = cv2.boundingRect(ctr)
            cv2.rectangle(new_ctr_image, (x,y), (x+w, y+h), (0,0, 255), 2)
            cv2.imwrite("visual_servoing/computer_vision/output_images/final_contours.png", new_ctr_image)

        x,y,w,h = cv2.boundingRect(np.vstack(contours_in_roi))
        bounding_box = ((x,y), (x+w, y+h))

        cv2.rectangle(new_ctr_image, bounding_box[0], bounding_box[1], (0,0,255), 2)
        cv2.imwrite("visual_servoing/computer_vision/output_images/final_bounds.png", new_ctr_image)

    return bounding_box

def get_image(img_num = None, img_type = "cone"):
    """
    Input:
        img_num: The number corresponding to the cone img

    Returns:
        The image corresponding to the given img number or a random cone image if no number is given
    """
    if img_type == "cone":
        test_imgs_dir = "./visual_servoing/computer_vision/test_images_cone"
    elif img_type == "line":
        test_imgs_dir = "./visual_servoing/computer_vision/test_images_line"

    if img_num is None:
        img_num = np.random.randint(1, 21)

    img_file = test_imgs_dir + "/test" + str(img_num) + ".jpg"
    return cv2.imread(img_file), img_num

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

def hsv_convert_to_cv2(hsv_input):
    hue, sat, val = hsv_input
    return (hue / 2, sat * 2.55, val * 2.55)

def get_hsv_range_by_colors(hsv_low, hsv_high):
    return {"lower": np.array(hsv_low), "upper": np.array(hsv_high)}

if __name__ == "__main__":
    img_type = "line"
    img, img_num = get_image(img_num = 2, img_type = img_type)
    print("Showing #", img_num)
    cv2.imwrite("visual_servoing/computer_vision/output_images/initial_img.png", img)

    cd_color_segmentation(img, detection_mode = img_type)