import numpy as np
import cv2

draw = False
force_max = True
N = 8000
THRESHOLD = 1e-4


def init_stipples(N, x_max, y_max):
    rX = np.random.randint(0, x_max, size=N).astype(dtype=np.float32)
    rY = np.random.randint(0, y_max, size=N).astype(dtype=np.float32)

    points = []
    for x, y in zip(rX, rY):
        points.append((x, y))
    return points

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] >= rect[2] :
        return False
    elif point[1] >= rect[3] :
        return False
    return True

def clamp_boundary(rect, point):
    """
    Clamps a point to the boundaries of the rectangle
    """
    point[0] = np.clip(point[0], rect[0], rect[2]-1)
    point[1] = np.clip(point[1], rect[1], rect[3]-1)
    return point

def lloyd_relaxation(image, N, draw=False):
    """
    Implements Lloyd relaxation algorithm on a color image. 
    Stipples are randomly initialized.
    """
    if np.ndim(image) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    # Initialize stipples
    points = init_stipples(N, image.shape[1], image.shape[0])
    rect = (0, 0, image.shape[1], image.shape[0])

    # Initialize Voronoi 
    subdiv = cv2.Subdiv2D(rect)
    subdiv.initDelaunay(rect)

    for p in points:
        subdiv.insert(p)

    iteration = 0
    d_facet_area_std = np.Inf
    prev_facet_area_std = 0
    learning_rate = 0.1

    while d_facet_area_std > THRESHOLD and iteration < 15:
        print("Starting iteration %d" % iteration)

        facet_area, facet_center, facet_contours = compute_contours(subdiv, rect, draw=draw)
        facet_area_std = np.std( facet_area[np.isfinite(facet_area)] )

        # Clear point list
        subdiv.initDelaunay(rect)
        new_pts = []
        
        # Recompute centers
        for c, p in zip(facet_center, points):
            new_pt = clamp_boundary(rect,  p + learning_rate*(c - p) )

            new_pts.append( new_pt )
            subdiv.insert( tuple(new_pt) )
           
        # Update recursion
        d_facet_area_std = np.abs(facet_area_std - prev_facet_area_std)
        prev_facet_area_std = facet_area_std
        
        print("\t stdev: %f" % d_facet_area_std)

        iteration += 1

    return facet_area, facet_center, facet_contours


    # cv2.rectangle(img_blank, (200, 200), (712, 712), (0, 255, 0), thickness=1)
    # cv2.imshow("test2", img_blank)
    # cv2.waitKey()
        # print()
def compute_contours(subdiv, rect, draw=False):
    # Iterate through each facet
    ( facets, centers) = subdiv.getVoronoiFacetList([])

    # Return values
    centroid_o = np.zeros(centers.shape, dtype=np.float32)
    area_o = np.zeros(centers.shape[0], dtype=np.float32)
    contour_o = []

    if draw:
        BORDER = 200
        img_blank = np.zeros(img.shape, dtype=np.uint8)
        img_blank = cv2.copyMakeBorder(img_blank, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, 0)

    # For each facet, compute its weighted centroid
    for i in range(0,len(facets)):
        ifacet_arr = []
        in_bounds = []
        for f in facets[i] :
            ifacet_arr.append(f)
            in_bounds.append(rect_contains(rect, f))

        ifacet = np.array(ifacet_arr, dtype=np.int)

        M = cv2.moments(ifacet)
        Mx = M["m10"] / (M["m00"]+1e-5)
        My = M["m01"] / (M["m00"]+1e-5)
        
        if all(in_bounds):
            area_o[i] = M["m00"]
        else:
            area_o[i] = np.NaN

        centroid_o[i] = (Mx, My)
        contour_o.append(ifacet)

        if draw:
            cv2.polylines(img_blank,  np.array([ifacet + BORDER], dtype=np.int32), True, (255, 255, 255), 1, cv2.LINE_4, 0)
            cv2.circle(img_blank, (int(np.round(Mx) + BORDER), int(np.round(My) + BORDER)), 1, (255, 0, 0), -1, cv2.LINE_8, 0)
    if draw:
        cv2.rectangle(img_blank, (rect[0] + BORDER, rect[1] + BORDER), (rect[2] + BORDER, rect[3] + BORDER), (0, 255, 0), thickness=1)
        cv2.imshow("voronoi", img_blank)
        cv2.waitKey(10)

    # End facet iteration

    return area_o, centroid_o, contour_o

def correct_edges(img, f_area, f_center, f_contours):
    rect = (0, 0, img.shape[1], img.shape[0])

    f_area_nan = np.isnan(f_area)
    f_center_bounded = [not rect_contains(rect, c) for c in f_center]
    f_area_nan_idx = np.argwhere(f_area_nan | f_center_bounded)

    for idx in f_area_nan_idx.flatten():
        f_a, f_c, f_ctr = f_area[idx], f_center[idx, :], f_contours[idx]

        img_mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(img_mask, f_ctr, (255, 255, 255), cv2.LINE_4, 0)
        img_masked = cv2.bitwise_and(img_gray, img_gray, mask=img_mask)

        M = cv2.moments(img_masked)
        Mx = M["m10"] / (M["m00"]+1e-5)
        My = M["m01"] / (M["m00"]+1e-5)

        f_area[idx] = M["m00"] * np.sqrt(2)
        f_center[idx] = (Mx, My)

        # if not rect_contains(rect, f_center[idx]):
        #     print()
    return f_area, f_center, f_contours
        

def draw_stipples(img, f_area, f_center, f_contours, force_max=False):

    # Create a copy of the image in grayscale
    if np.ndim(img) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    if force_max:
        img_canvas = np.zeros(img.shape, dtype=np.float32)
    else:
        img_canvas = np.zeros(img.shape, dtype=np.float32) + 255
    rect = (0, 0, img.shape[1], img.shape[0])
    # Compute the maximum radius for each facet
    max_radius = np.sqrt(f_area / np.pi)

    grad_mag, grad_ang = image_gradient(img_gray)

    # Iterate through each set of contours
    # HACKY: Create a bitmask using each contours and compute the weight of the image relative to the area
    # img_blank = np.zeros(img.shape, dtype=np.float32)
    for f_a, f_c, f_ctr, R in zip(f_area, f_center, f_contours, max_radius):
        # Edge region
        if np.isnan(f_a) or not rect_contains(rect, f_c):
            continue

        img_mask = np.zeros(img_gray.shape, dtype=np.uint8) # img_blank.copy()

        # Create the mask
        cv2.fillConvexPoly(img_mask, f_ctr, (255, 255, 255), cv2.LINE_4, 0)

        # Mask the image and compute the centroid
        img_masked = cv2.bitwise_and(img_gray, img_gray, mask=img_mask)

        maxIdx = np.argmax( cv2.bitwise_and(grad_mag, grad_mag, mask=img_mask) )
        maxMag = grad_mag.flatten()[maxIdx]
        maxAng = grad_ang.flatten()[maxIdx] * 360

        # Nearest neighbor ccolor
        ptColor = img[int(f_c[1]), int(f_c[0])].astype(np.double)
        if ptColor.size > 1:
            ptColor = tuple(ptColor)

        # Compute the momentimg[]
        M = cv2.moments(img_masked)

        
        stipple_radius = (1 - np.clip(M["m00"] / (f_a + 1e-5), 0, 1)) * R
        stipple_center = tuple(np.array([f_c[0], f_c[1]], dtype=np.float32))
        # Circular stipples
        # cv2.circle(img_canvas, tuple(f_c.astype(np.int32)), int(stipple_radius), (0, 0, 0), -1, cv2.LINE_AA)

        # Elliptical stipples
        # if maxMag != 0:

        if not force_max:
            axesLength = np.array([stipple_radius * (1+maxMag), stipple_radius], dtype=np.uint8) 
            # else:
                # axesLength = np.array([stipple_radius, stipple_radius], dtype=np.uint8) 
            cv2.ellipse(img_canvas, stipple_center, tuple(axesLength), 
                angle=maxAng, startAngle=0, endAngle=360, 
                color=ptColor, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img_canvas, stipple_center, int(R * 0.8), ptColor, -1, cv2.LINE_AA)

    return img_canvas

def image_gradient(img):
    # Compute the gradient of an image
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    mag_sobel = cv2.magnitude(sobelx64f, sobely64f) 
    mag_sobel /= np.max(mag_sobel)

    ang_sobel = cv2.phase( sobelx64f, sobely64f ) / (2 * np.pi)

    # cv2.imshow("1", mag_sobel)
    # cv2.imshow("2", ang_sobel)
    # cv2.waitKey()

    return mag_sobel.astype(dtype=np.float32), ang_sobel.astype(dtype=np.float32)

def draw_voronoi(img, facets, centers) :
    BORDER = 200
    img_blank = np.zeros(img.shape, dtype=np.uint8)
    img_blank = cv2.copyMakeBorder(img_blank, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, 0)
    rect = (0, 0, img.shape[1], img.shape[0])

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
        
        ifacet = np.array(ifacet_arr, np.int)

        color_g = 0
        color = (color_g, color_g, color_g)

        border_color = (255, 255, 255)

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])

        cv2.polylines(img_blank,  np.array([ifacet + BORDER], dtype=np.int32), True, (255, 255, 255), 1, cv2.LINE_4, 0)
        cv2.circle(img_blank, (int(np.round(centers[i][0]) + BORDER), int(np.round(centers[i][1]) + BORDER)), 1, (255, 0, 0), -1, cv2.LINE_8, 0)
    cv2.rectangle(img_blank, (rect[0] + BORDER, rect[1] + BORDER), (rect[2] + BORDER, rect[3] + BORDER), (0, 255, 0), thickness=1)
    cv2.imshow("tlwnbsalg", img_blank)
    cv2.waitKey()


if __name__ == "__main__":
    
    # img = cv2.imread("snowman_scene.png", cv2.IMREAD_COLOR)
    img = cv2.imread("flowers.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = img.astype(dtype=np.float32) / 255 # Convert to float

    # img = np.moveaxis(img, [0, 1, 2], [1, 0, 2])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blue = img[:, :, 0]
    img_green = img[:, :, 1]
    img_red = img[:, :, 2]

    print(img.shape)

    # facet_area, facet_center, facet_contours = lloyd_relaxation(img, N, draw=draw)
    # canvas_all = draw_stipples(img, facet_area, facet_center, facet_contours)
    print("Performing Lloyd relaxation...")
    facet_area, facet_center, facet_contours = lloyd_relaxation(img_gray, N, draw=draw)

    print("Correcting edges...")
    facet_area, facet_center, facet_contours = correct_edges(img_gray,  facet_area, facet_center, facet_contours)
    # img_blank = np.zeros(img.shape, img.dtype)
    # draw_voronoi(img_blank, facet_contours, facet_center)


    print("Generating channel-wise stipples...")
    print("\t Blue")
    # facet_area, facet_center, facet_contours = lloyd_relaxation(img_blue, N, draw=draw)
    canvas_blue = draw_stipples(img_blue, facet_area, facet_center, facet_contours, force_max=force_max)

    print("\t Green")
    # facet_area, facet_center, facet_contours = lloyd_relaxation(img_green, N, draw=draw)
    canvas_green = draw_stipples(img_green, facet_area, facet_center, facet_contours, force_max=force_max)

    print("\t Red")
    # facet_area, facet_center, facet_contours = lloyd_relaxation(img_red, N, draw=draw)
    canvas_red = draw_stipples(img_red, facet_area, facet_center, facet_contours, force_max=force_max)

    # img_composite = np.zeros(img.shape, dtype=img.dtype)

    canvas_comb = np.stack([canvas_blue, canvas_green, canvas_red], axis=2)
    canvas_comb *= 255
    cv2.imwrite("file.png", canvas_comb.astype(np.uint8))    
    # cv2.imshow("original", img)
    # # cv2.imshow("gray", canvas_all)
    # cv2.imshow("blue", canvas_blue)
    # cv2.imshow("green", canvas_green)
    # cv2.imshow("red", canvas_red)
    # cv2.imshow("combine", canvas_comb)
    # cv2.waitKey()

    # print()
