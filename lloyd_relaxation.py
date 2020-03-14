import numpy as np
import cv2

class lloyd_relaxation():
    def __init__(self, image, N=1000, threshold=1e-3, maxiter=15, grad=False, dots=False):
        self.N = N
        self.threshold = threshold
        self.maxiter = maxiter
        self.grad = grad
        self.img = image
        self.dots = dots
        self.last_run_info = ()

    def run_all(self):
        img = self.img
        N = self.N
        grad = self.grad

        print("Performing Lloyd relaxation...")
        facet_area, facet_center, facet_contours, region_ids = self.lloyd_relaxation(img, N)

        print("Correcting edges...")
        facet_area, facet_center, facet_contours = self.correct_edges(img, facet_area, facet_center, facet_contours)

        self.last_run_info = (facet_area, facet_center, facet_contours, region_ids)

        print("Generating stipples...")
        canvas_comb = self.draw_stipples_all(img, facet_area, facet_center, facet_contours, region_ids)

        print("done")

        return (canvas_comb*255).astype(np.uint8)

    def run_style(self):
        img = self.img
        N = self.N
        grad = self.grad

        facet_area, facet_center, facet_contours, region_ids = self.last_run_info

        print("Generating stipples...")
        canvas_comb = self.draw_stipples_all(img, facet_area, facet_center, facet_contours, region_ids)

        print("done")
        return (canvas_comb*255).astype(np.uint8)

    def init_stipples(self, N, x_max, y_max):
        rX = np.random.randint(0, x_max, size=N).astype(dtype=np.float32)
        rY = np.random.randint(0, y_max, size=N).astype(dtype=np.float32)

        points = []
        for x, y in zip(rX, rY):
            points.append((x, y))
        return points

    # Check if a point is inside a rectangle
    def rect_contains(self, rect, point) :
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] >= rect[2] :
            return False
        elif point[1] >= rect[3] :
            return False
        return True

    def clamp_boundary(self, rect, point):
        """
        Clamps a point to the boundaries of the rectangle
        """
        point[0] = np.clip(point[0], rect[0], rect[2]-1)
        point[1] = np.clip(point[1], rect[1], rect[3]-1)
        return point

    def lloyd_relaxation(self, image, N, draw=False):
        """
        Implements Lloyd relaxation algorithm on a color image. 
        Stipples are randomly initialized.
        """
        if np.ndim(image) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        # Initialize stipples
        points = self.init_stipples(N, image.shape[1], image.shape[0])
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

        while d_facet_area_std > self.threshold and iteration < self.maxiter:
            print("Starting iteration %d" % iteration)

            facet_area, facet_center, facet_contours, region_ids = self.compute_contours(subdiv, rect, draw=draw)
            facet_area_std = np.std( facet_area[np.isfinite(facet_area)] )

            # Clear point list
            subdiv.initDelaunay(rect)
            new_pts = []
            
            # Recompute centers
            for c, p in zip(facet_center, points):
                new_pt = self.clamp_boundary(rect,  p + learning_rate*(c - p) )

                new_pts.append( new_pt )
                subdiv.insert( tuple(new_pt) )
            
            # Update recursion
            d_facet_area_std = np.abs(facet_area_std - prev_facet_area_std)
            prev_facet_area_std = facet_area_std
            
            print("\t stdev: %f" % d_facet_area_std)

            iteration += 1

        return facet_area, facet_center, facet_contours, region_ids


    def compute_contours(self, subdiv, rect, draw=False):
        # Iterate through each facet
        ( facets, centers) = subdiv.getVoronoiFacetList([])

        # Return values
        centroid_o = np.zeros(centers.shape, dtype=np.float32)
        area_o = np.zeros(centers.shape[0], dtype=np.float32)
        contour_o = []

        region_ids = np.zeros((rect[3], rect[2]), dtype=np.int32)

        # For each facet, compute its weighted centroid
        for i in range(0,len(facets)):
            ifacet_arr = []
            in_bounds = []
            for f in facets[i] :
                ifacet_arr.append(f)
                in_bounds.append(self.rect_contains(rect, f))

            ifacet = np.array(ifacet_arr, dtype=np.int)

            cv2.fillConvexPoly(region_ids, ifacet, i, cv2.LINE_4, 0)

            M = cv2.moments(ifacet)
            Mx = M["m10"] / (M["m00"]+1e-5)
            My = M["m01"] / (M["m00"]+1e-5)
            
            if all(in_bounds):
                area_o[i] = M["m00"]
            else:
                area_o[i] = np.NaN

            centroid_o[i] = (Mx, My)
            contour_o.append(ifacet)

        # End facet iteration

        return area_o, centroid_o, contour_o, region_ids

    def correct_edges(self, img, f_area, f_center, f_contours):
        if np.ndim(img) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        rect = (0, 0, img.shape[1], img.shape[0])

        f_area_nan = np.isnan(f_area)
        f_center_bounded = [not self.rect_contains(rect, c) for c in f_center]
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
            

    def draw_stipples(self, img, f_area, f_center, f_contours, region_ids, force_max=False):

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

        if not force_max:
            grad_mag, grad_ang = image_gradient(img_gray)

        # Iterate through each set of contours
        # HACKY: Create a bitmask using each contours and compute the weight of the image relative to the area
        # img_blank = np.zeros(img.shape, dtype=np.float32)
        for f_a, f_c, f_ctr, R in zip(f_area, f_center, f_contours, max_radius):
            # t0 = time.time()

            # Edge region
            if np.isnan(f_a) or not rect_contains(rect, f_c):
                continue

            img_mask = np.zeros(img_gray.shape, dtype=np.uint8) # img_blank.copy()

            # Create the mask
            cv2.fillConvexPoly(img_mask, f_ctr, (255, 255, 255), cv2.LINE_4, 0)

            # Mask the image and compute the centroid
            img_masked = cv2.bitwise_and(img_gray, img_gray, mask=img_mask)

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
                maxIdx = np.argmax( cv2.bitwise_and(grad_mag, grad_mag, mask=img_mask) )
                maxMag = grad_mag.flatten()[maxIdx]
                maxAng = grad_ang.flatten()[maxIdx] * 360

                axesLength = np.array([stipple_radius * (1+maxMag), stipple_radius], dtype=np.uint8) 
                # else:
                    # axesLength = np.array([stipple_radius, stipple_radius], dtype=np.uint8) 
                cv2.ellipse(img_canvas, stipple_center, tuple(axesLength), 
                    angle=maxAng, startAngle=0, endAngle=360, 
                    color=ptColor, thickness=-1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(img_canvas, stipple_center, int(R * 0.8), ptColor, -1, cv2.LINE_AA)
            # t1 = time.time()
            # print( t1-t0 )

        return img_canvas

    def image_gradient(self, img):
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

    def draw_stipples_all(self, img, f_area, f_center, f_contours, region_ids):
        # Create a copy of the image in grayscale
        assert( np.ndim(img) == 3 )

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_canvas = np.zeros(img.shape, dtype=np.float32)
        if self.dots:
            img_canvas += 1

        rect = (0, 0, img.shape[1], img.shape[0])
        # Compute the maximum radius for each facet
        max_radius = np.sqrt(f_area / np.pi)

        if self.grad:
            grad_mag, grad_ang = self.image_gradient(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        for i, z in enumerate(zip(f_area, f_center, f_contours, max_radius)):
            f_a, f_c, f_ctr, R = z
            # t0 = time.time()

            # Edge region
            if np.isnan(f_a) or not self.rect_contains(rect, f_c):
                continue
            
            # Nearest neighbor ccolor
            ptColor = tuple(img[int(f_c[1]), int(f_c[0]), :].astype(np.double))
            stipple_center = tuple(np.array([f_c[0], f_c[1]], dtype=np.float32))

            # Dots are proportional to the intensity / area of the region
            if self.dots:
                intensity_ratio = np.sum(1 - img_gray[region_ids == i]) / f_a
                ptColor = (0, 0, 0)
                R = np.max([R*intensity_ratio, 2]) / 0.8

            maxAng = 0
            axesLength = np.array([R * 0.8, R * 0.8], dtype=np.uint8) 
            
            if self.grad:
                mag_rvals = grad_mag[region_ids == i]
                ang_rvals = grad_ang[region_ids == i]
                
                if mag_rvals.size != 0:
                    maxIdx = np.argmax(mag_rvals)

                    maxMag = mag_rvals[maxIdx]
                    maxAng = ang_rvals[maxIdx] * 360

                    hAx = R * (1 + maxMag / 3) * 0.8
                    vAx = R * (1 - maxMag / 3) * 0.8
                    axesLength = np.array([hAx, vAx], dtype=np.uint8) 

            # Circular stipples
            # cv2.circle(img_canvas, stipple_center, int(R * 0.8), ptColor, -1, cv2.LINE_AA)

            cv2.ellipse(img_canvas, stipple_center, tuple(axesLength), 
                    angle=maxAng, startAngle=0, endAngle=360, 
                    color=ptColor, thickness=-1, lineType=cv2.LINE_AA)
            # t1 = time.time()
            # print( t1-t0 )
        return img_canvas

    def draw_voronoi(self, img, facets, centers) :

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
    path = "images/plant.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).astype(dtype=np.float32) / 255 # Convert to float
    L = lloyd_relaxation(img, N=10000, dots=True)
    img_out = L.run_all()
    cv2.imwrite("file%d.png" % L.N, img_out)

