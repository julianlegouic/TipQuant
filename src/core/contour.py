import cv2 as cv
import numpy as np

from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
from shapely.geometry.multipoint import MultiPoint

from src.core.utils import get_mask, adjust_gamma
from src.utils import get_contour_ring


class ContourDetection:
    """
    From a gray frame detects a contour as a set of points.
    Points are ordered spatially.
    Contour is represented by a N x 2 array. With N being the number of contour points.
    """

    def __init__(self, ksize=None, sigma=None, gamma=None, keep_mask=None):
        """
        :param ksize: size of kernels for smoothing operations
        :param sigma: sigma of the gaussian blur
        :param gamma: gamma coefficient for the gamma correction
        :param keep_mask: boolean flag to tell if mask need to be kept
            during contour detection
        """
        self.ksize = (ksize, ksize)
        self.sigma = sigma
        self.gamma = gamma
        self.keep_mask = keep_mask
        self.thresh = None
        self.computed_thresh = []
        self.max_erosion = 10

    @classmethod
    def from_config(cls, config):
        ksize = config["CONTOUR"]["KERNEL_SIZE"]
        sigma = config["CONTOUR"]["SIGMA"]
        gamma = config["CONTOUR"]["GAMMA"]
        if config["CONTOUR"]["MASK"]:
            keep_mask = config["CONTOUR"]["MASK"]
        else:
            keep_mask = False
        return cls(ksize=ksize, sigma=sigma, gamma=gamma, keep_mask=keep_mask)

    def _preprocess(self, frame):
        """
        apply smoothing operations on the frame so that it is easier to find the principal contour
        :param frame: input frame
        :return: processed frame
        """
        frame = adjust_gamma(frame, gamma=self.gamma)
        frame = cv.GaussianBlur(frame, ksize=self.ksize, sigmaX=self.sigma)
        _, frame = cv.threshold(frame, self.thresh, 255, cv.THRESH_BINARY)
        frame = self._fill_contour(frame)
        return frame

    @staticmethod
    def _find_contours(frame):
        """
        find all the contours in the given frame and return the contour with max area
        :param frame: input frame
        :return: all the contours detected in the frame
        """
        contours, _ = cv.findContours(frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x))
        contour = contours[-1]
        contour = contour[:, 0, :]
        return contour

    @staticmethod
    def _contour_union(cnt1, cnt2, shape):
        """
        from two contours, compute the union contour (fused for non regression)
        :param cnt1: contour 1
        :param cnt2: contour 2
        :param shape: shape of the original frame
        :return: union of contours cnt1 and cnt2
        """
        cnt1_mask = get_mask(cnt1, shape, "fill")
        cnt2_mask = get_mask(cnt2, shape, "fill")
        joined_mask = cv.bitwise_or(cnt1_mask, cnt2_mask)
        contours, _ = cv.findContours(joined_mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        return contours[0][:, 0, :]

    @staticmethod
    def _average_contour(cnt1, cnt2, shape):
        """
        from two contours, compute the average contour (fused for non regression)
        :param cnt1: contour 1
        :param cnt2: contour 2
        :param shape: shape of the original frame
        :return: average contour between cnt1 and cnt2
        """
        cnt1_mask = get_mask(cnt1, shape, "fill")
        cnt2_mask = get_mask(cnt2, shape, "fill")
        dt1 = cv.distanceTransform(cnt1_mask, cv.DIST_L2, 3)
        dt1_inv = cv.distanceTransform(~cnt1_mask, cv.DIST_L2, 3)
        dt2 = cv.distanceTransform(cnt2_mask, cv.DIST_L2, 3)
        dt2_inv = cv.distanceTransform(~cnt2_mask, cv.DIST_L2, 3)
        mask = ((dt1 - dt1_inv) + (dt2 - dt2_inv) > 0).astype(np.uint8)
        contours, _ = cv.findContours(mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        return contours[0][:, 0, :]

    def _fit_contour(self, video_frame, mask):
        """
        compute mean value of pixels on the contour a higher value indicates
        a better fitted contour
        :param video_frame: frame of the original video
        :param mask: detected mask of the frame
        :return: mean value of contour's pixels
        """
        contour = self._find_contours(mask)
        mean = 0
        for c in contour:
            y, x = c
            mean += video_frame[x, y]
        return mean / len(contour)

    def _adjust_contour(self, video_frame, mask):
        """
        apply erosion to have a better fit of the tube shape if necessary
        :param video_frame: frame of the video
        :param mask: detected mask of the frame
        :return: adjusted mask
        """
        max_mean = self._fit_contour(video_frame, mask)

        i, gap, gap_inc = 0, 0, True
        while gap_inc and i < self.max_erosion:
            gap_inc = False
            mean = self._fit_contour(video_frame, cv.erode(mask, np.ones((3, 3), np.uint8), iterations=i+1))
            if mean > max_mean:
                gap_tmp = mean - max_mean
                max_mean = mean
                if (gap == 0 and gap_tmp > 5) or (gap != 0 and gap_tmp > gap*2/3):
                    gap = gap_tmp
                    gap_inc = True
                    i += 1
        if i > 0:
            mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=i)
            # set a maximum number of erosion for the rest of the video
            if (self.max_erosion == 10) and (not self.keep_mask):
                self.max_erosion = i
        return mask

    def _fill_contour(self, frame):
        """
        take the biggest contour on the image and fill it

        Note: improvements can be done to fill a U-shaped
        pollen tube. Currently not supported.

        :param frame: current processed frame
        :return: mask of the biggest filled contour
        """
        h, w = frame.shape
        contour = self._find_contours(frame)
        min_x_bounds, min_y_bounds = np.min(contour, axis=0)
        max_x_bounds, max_y_bounds = np.max(contour, axis=0)
        new_frame = np.zeros_like(frame)
        nb_pix_left = len(np.where(contour[:, 0] == 0)[0])
        nb_pix_right = len(np.where(contour[:, 0] == (w - 1))[0])
        nb_pix_top = len(np.where(contour[:, 1] == 0)[0])
        nb_pix_bottom = len(np.where(contour[:, 1] == (h - 1))[0])
        orientation = np.argmax([
            max(nb_pix_left, nb_pix_right),
            max(nb_pix_top, nb_pix_bottom)
        ])
        # process the image in the orientation of the tube (mostly max(w, h))
        if orientation == 0:
            # growing horizontally
            for x in range(min_x_bounds, max_x_bounds+1):
                x_idx = np.where(contour[:, 0] == x)[0]
                y_min = np.min(contour[x_idx][:, 1])
                y_max = np.max(contour[x_idx][:, 1])
                for y in range(y_min, y_max+1):
                    new_frame[y, x] = 255
        else:
            # growing vertically
            for y in range(min_y_bounds, max_y_bounds+1):
                y_idx = np.where(contour[:, 1] == y)[0]
                x_min = np.min(contour[y_idx][:, 0])
                x_max = np.max(contour[y_idx][:, 0])
                for x in range(x_min, x_max+1):
                    new_frame[y, x] = 255
        return new_frame

    def detect_contour(self, frame, prev_contour=None, preprocess=True):
        """
        regroups all the steps that get a contour from a frame
        :param frame: input frame
        :param prev_contour: previous contour (for non regression)
        :param preprocess: whether we use preprocessing on the frame or not
        :return: principal contour of the frame
        """
        if preprocess:
            processed_frame = self._preprocess(frame)
        contour = self._find_contours(processed_frame)
        if prev_contour is not None:
            if not self.keep_mask:
                adjusted_mask = self._adjust_contour(frame, processed_frame)
                adjusted_contour = self._find_contours(adjusted_mask)
                average_contour = self._average_contour(adjusted_contour, prev_contour, frame.shape)
                avg_mask = get_mask(average_contour, frame.shape, "fill")
                prev_mask = get_mask(prev_contour, frame.shape, "fill")
                if np.max(np.subtract(avg_mask, prev_mask)) == 255:
                    # Average contour is larger than previous contour
                    return average_contour
                elif np.max(np.subtract(adjusted_mask, prev_mask)) != 255:
                    # Normal contour is smaller than previous contour
                    return self._contour_union(adjusted_contour, prev_contour, frame.shape)
            else:
                contour_union = self._contour_union(contour, prev_contour, frame.shape)
                contour_mask = get_mask(contour_union, frame.shape, "fill")
                adjusted_mask = self._adjust_contour(frame, contour_mask)
                adjusted_contour = self._find_contours(adjusted_mask)
        else:
            adjusted_mask = self._adjust_contour(frame, processed_frame)
            adjusted_contour = self._find_contours(adjusted_mask)
        return adjusted_contour

    def get_threshold(self, frames):
        """
        Set global minimum threshold for preprocessing
        :param frames: list of gray frames
        :return:
        """
        for frame in frames:
            frame = adjust_gamma(frame, gamma=self.gamma)
            frame = cv.GaussianBlur(frame, ksize=self.ksize, sigmaX=self.sigma)
            if self.thresh is None:
                self.thresh, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                self.computed_thresh.append(self.thresh)
            else:
                new_thresh, _ = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                self.computed_thresh.append(new_thresh)
                if new_thresh < self.thresh:
                    self.thresh = new_thresh


class ContourParameterization:

    """ From a contour as a set of points, parameterizes it with splines. """

    def __init__(self, degree, knots_ratio):
        """
        :param degree: degree of the splines
        :param knots_ratio: number of knots if int else percentage of contour used as knots
        """
        self.degree = degree
        self.knots_ratio = knots_ratio

    @classmethod
    def from_config(cls, config):
        degree = config["CONTOUR"]["SPLINE_DEGREE"]
        knots_ratio = config["CONTOUR"]["SPLINE_KNOTS_RATIO"]
        knots_ratio = int(knots_ratio) if knots_ratio > 1 else knots_ratio
        return cls(degree=degree, knots_ratio=knots_ratio)

    def _fit_spline(self, contour):
        """
        from contour points, fit splines and returns its parameters and latent coordinate
        :param contour: contour points
        :return: splines parameters (tck), evaluation points (us)
        """
        xs, ys = contour.T
        xs, ys = xs.tolist(), ys.tolist()
        n_knots = int(len(xs) * self.knots_ratio) if isinstance(self.knots_ratio, float) \
            else self.knots_ratio
        t = [0.0, 0.0, 0.0] + list(np.linspace(0, 1, n_knots)) + [1.0, 1.0, 1.0]
        tck, us = splprep([xs, ys], t=t, k=self.degree, task=-1, quiet=5)
        return tck, us

    @staticmethod
    def _interpolate(tck, umin, umax, npoints, shape):
        """
        from splines, computes the contour points between umin and umax with npoints sampling
        :param tck: splines parameters
        :param umin: evaluation abs start
        :param umax: evaluation abs end
        :param npoints: number of points to evaluate between umin and umax
        :param shape: shape of the frame (ensure border constraints)
        :return: contour given by splines, abs of computation
        """
        us_new = np.linspace(umin, umax, npoints)
        xs_new, ys_new = splev(us_new, tck, der=0)

        res_array, us = list(), list()
        x_prev, y_prev = -1, -1  # ensure uniqueness of points
        for x, y, u in zip(xs_new, ys_new, us_new):
            x, y = round(x), round(y)
            if (x != x_prev or y != y_prev) and 0 <= x < shape[1] and 0 <= y < shape[0]:
                us.append(u)
                res_array.append([x, y])
                x_prev, y_prev = x, y
        contour = np.asarray(res_array, dtype=np.int32)
        return contour, us

    def parameterize_contour(self, contour, shape):
        """
        general function that takes a contour and return splines parameters, spline smoothed contour
        as well as latent abscissa
        :param contour: contour points
        :param shape: original frame shape
        :return: contour interpolated points, splines parameters, spline abscissa
        """
        tck, us = self._fit_spline(contour)
        contour, us = self._interpolate(tck, us.min(), us.max(), len(contour), shape)
        return contour, tck, us


class ContourCharacterization:

    """ From a parameterized contour, computes characteristics such as tangents, normals or
     curvatures. """

    def __init__(self):
        pass

    @staticmethod
    def _normalize(array):
        """
        normalizes vectors in an array of size n_vectors x 2
        :param array: array of vectors
        :return: normalized array
        """
        f = lambda vector: vector / np.linalg.norm(vector)
        normed = np.apply_along_axis(f, axis=1, arr=array)
        return normed

    @staticmethod
    def _compute_spline_derivatives(tck, us, order):
        """
        from splines coefficients and evaluation points, compute its derivatives
        :param tck: splines coefficients
        :param us: evaluation points
        :param order: order of the derivatives
        :return: derivatives evaluated at each evaluation point
        """
        dXs, dYs = splev(us, tck, der=order)
        dS = list()
        for dx, dy in zip(dXs, dYs):
            dS.append([dx, dy])
        dS = np.array(dS)
        return dS

    def _get_tangents(self, tck, us):
        """
        computes tangents as defined by the first order normalized derivatives of the surface
        :param tck: spline coefficients
        :param us: evaluation points
        :return: array of vectors of size n_evaluation_points x 2
        """
        dS = self._compute_spline_derivatives(tck, us, order=1)
        T = self._normalize(dS)
        return T

    def _get_normals(self, T):
        """
        computes normals from tangents as their rotated 90 degree vector of unit 1
        :param T: tangents
        :return: normals as array of size n_tangents x 2
        """
        N = list()
        for t_vect in T:
            tx, ty = t_vect[0], t_vect[1]
            N.append([-ty, tx])  # unormalized rotated vector
        N = np.array(N)
        N = self._normalize(N)
        return N

    @staticmethod
    def _get_curvatures(T, us):
        """
        computes curvatures as the norm of the derivatives of the tangent vector (by finite
        differences)
        :param T: tangents
        :param us: delta between tangent points
        :return: curvatures
        """
        gradient = lambda x: np.gradient(x, us)
        dT = np.apply_along_axis(gradient, axis=0, arr=T)
        K = np.apply_along_axis(np.linalg.norm, axis=1, arr=dT)
        return K

    def characterize_contour(self, tck, us):
        """
        from a contour parameterized by splines (tck, us), compute at each point, tangents,
        normals and curvatures
        :param tck: splines coefficients
        :param us: splines evaluation points
        :return: tangents, normals, curvatures
        """
        T = self._get_tangents(tck, us)
        N = self._get_normals(T)
        curvatures = self._get_curvatures(T, us)
        return T, N, curvatures


class ContourROI:

    """ This class works with two contours, an origin and one from the future. By computing the
    difference of the two, we locate the area of growth and define it as our region of interest.
    """

    def __init__(self):
        pass

    @staticmethod
    def _select_contour(contours):
        """
        selects the contour with maximum area from a list of contours, removes the fake dim
        :param contours: list of contours
        :return: max area contour
        """
        contours = sorted(contours, key=lambda x: cv.contourArea(x))
        contour = contours[-1]
        contour = contour[:, 0, :]
        return contour

    def _get_diff_contour(self, current_contour, next_contour, shape):
        """
        find the contour of the region that has grown from current contour to next contour.
        :param current_contour: current contour points
        :param next_contour: contour points in the future
        :param shape: original shape of the frame
        :return: contour points of the growth region
        """
        curr_mask = get_mask(current_contour, shape, "fill")
        next_mask = get_mask(next_contour, shape, "fill")

        diff = np.subtract(next_mask, curr_mask)
        _, diff = cv.threshold(diff, 2, 255, cv.THRESH_BINARY)
        dilated_1 = cv.dilate(diff, kernel=np.ones((3, 3), np.uint8), iterations=1)

        growth_contours, _ = cv.findContours(dilated_1, mode=cv.RETR_TREE,
                                             method=cv.CHAIN_APPROX_NONE)
        growth_contour = self._select_contour(growth_contours)
        growth_mask = get_mask(growth_contour, shape, "fill")
        growth_mask = cv.dilate(growth_mask, kernel=np.ones((3, 3), np.uint8), iterations=2)
        contour = cv.bitwise_and(get_mask(current_contour, shape, "lines", isClosed=False),
                                 growth_mask)
        contour = cv.findNonZero(contour)
        return contour[:, 0, :]

    @staticmethod
    def _select_points(diff_contour, current_contour):
        """
        finds the region in the current contour that has grown to the next contour
        :param diff_contour: contour of the difference between contours at different timesteps
        :param current_contour: current contour
        :return: contour points from the current contour that have shown growth
        """
        roi_indices = []
        for point in diff_contour:
            index = np.where((current_contour == point).all(1))
            if len(index[0]) > 0:
                roi_indices.append(index[0][0])
        return roi_indices

    def get_roi(self, current_contour, next_contour, shape):
        """
        from the current contour and a next contour find the contour points from the current contour
        that have shown growth
        :param current_contour: current contour
        :param next_contour: next contour
        :param shape: original shape
        :return: contour points from the current contour that have shown growth
        """
        diff_contour = self._get_diff_contour(current_contour, next_contour, shape)
        roi_indices = self._select_points(diff_contour, current_contour)
        return roi_indices


class ContourDisplacement:

    """ Computes the displacement of the contour between two frames wrt the intersection of the
    normals of the first contour to the second contour """

    def __init__(self):
        pass

    @staticmethod
    def _get_contour_line(contour):
        """
        we use shapely to compute intersections so we transform the contour as an array to
        a shapely linestring
        :param contour: contour points
        :return: shapely line string corresponding to the contour
        """
        return LineString(contour)

    @staticmethod
    def _get_normal_line(origin, normal, shape):
        """
        convenience function to get a shapely linestring from a point and its normal vectors
        :param origin: origin point
        :param normal: normal vector
        :param shape: shape of the image (to know what is infinity)
        :return: shapely line string defined by a point and its normal
        """
        end = origin + normal * (shape[0] * shape[1])  # end point to "infinity"
        return LineString([
            (origin[0], origin[1]),
            (end[0], end[1])
        ])

    def get_displacements(self, current_contour, next_contour, current_normals, roi_indices, shape):
        """
        for each point in the current contour belonging to the roi, we find the intersection of
        its normal vector to the next contour and compute the distance to the intersection point.
        In case of multiple intersections (curvy contour), we keep the minimum displacement.
        :param current_contour: current contour
        :param next_contour: contour in the future
        :param current_normals: normals of the current contour
        :param roi_indices: indices of the points in the region of interest
        :param shape: original shape
        :return:
        """
        next_contour_ring = get_contour_ring(next_contour)
        displacements = []

        for i in range(len(current_contour)):
            displacement = 0
            # if current index in roi_indices compute displacement at the point
            # else add 0 for non-roi indices
            if i in roi_indices:
                origin = current_contour[i]
                normal = current_normals[i]
                normal_line = self._get_normal_line(origin, normal, shape)
                if normal_line.intersects(next_contour_ring):
                    intersec = normal_line.intersection(next_contour_ring)
                    if isinstance(intersec, MultiPoint):
                        displs = [np.linalg.norm(np.array([point.x, point.y]) - origin)
                                  for point in intersec]
                        displacement = min(displs)
                    else:
                        intersec = np.array([intersec.x, intersec.y])
                        displacement = np.linalg.norm(intersec - origin)
            displacements.append(displacement)

        return np.array(displacements)

    @staticmethod
    def get_direction(displacements, normals):
        """
        compute the mean direction: the direction of all normals weighted by their displacement.
        The displacements are filtered so that outliers are removed (it can happen that a normal
        is misestimated).
        :param displacements: displacements
        :param normals: normals
        :return: weighted average direction
        """
        direction = np.array([0, 0], dtype=np.float64)
        if displacements.sum() > 0:
            quants = np.quantile(displacements[displacements != 0], [0.25, 0.75])
            delta = quants[1] - quants[0]
            for displacement, normal in zip(displacements, normals):
                if displacement <= quants[1] + 1.5 * delta:
                    direction += displacement * normal
        return direction


def project_on_contour(contour, point):
    """
    finds the index of the projection of a point to the contour
    :param contour: contour points as array of size n_points x 2
    :param point: point to be projected as array of size 2 x 1
    :return: index of the projected point on the contour
    """
    min_dist, index = float("inf"), 0
    for cnt_index, cnt_point in enumerate(contour):
        dist = np.linalg.norm(point - cnt_point)
        if dist < min_dist:
            min_dist, index = dist, cnt_index
    return index
