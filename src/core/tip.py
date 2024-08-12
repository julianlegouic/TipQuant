import statistics

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.core.contour import project_on_contour
from src.utils import compute_weights_df


class TipDetection:

    """ From the contour and its displacements or normals, this class finds the tip as defined by
    the point of maximal growth. """

    def __init__(self, window_size):
        """
        :param window_size: size of the window for averaging contour properties
        """
        self.window_size = window_size

    @classmethod
    def from_config(cls, config):
        """ creates a class instance from config dict """
        window_size = config["TIP"]["WINDOW_SIZE"]
        return cls(window_size)

    def _get_window_indices(self, window_center_index, max_index):
        """
        gets indices in a list of points around a window center (% used for edge cases)
        i.e. if we have [1, 2, 3, 4 , 5, 6] and the center is 6 (index 5) with window size 2,
        the function would return [0, 4, 5].
        :param window_center_index: center index of the window
        :param max_index: max index of the list
        :return: list of indices belonging to the window
        """
        half_window = int(self.window_size / 2)
        return [index % max_index for index in
                range(window_center_index - half_window, window_center_index + half_window + 1)]

    def get_tips_from_displacements(self, displacements, roi_indices):
        """
        finds the tip by selecting the point with maximal displacement (local growth) over a window.
        We compute the median over the window to cancel out the noise.
        :param displacements: local displacements of each point of the contour
        :param roi_indices: indices of the points of interest of the contour
        :return: index of the tip wrt the contour
        """
        max_index = len(displacements)
        max_displacement = 0
        tip_index = roi_indices[0]

        for roi_index in roi_indices:
            window_indices = self._get_window_indices(roi_index, max_index)
            displacement = statistics.median([displacements[i] for i in window_indices])
            if displacement > max_displacement:
                max_displacement = displacement
                tip_index = roi_index

        return tip_index

    def get_tips_from_normals(self, normals, displacements, direction, roi_indices, prev_tip, contour):
        """
        finds the tip by comparing normal directions of the possible
        tip candidates and the global growth direction
        :param normals: normal vectors at each contour point
        :param direction: growth direction vector
        :param roi_indices: indices of the points of interest of the contour
        :param prev_tip: tip location of the previous frame
        :param contour: contour of the current frame
        :return: index of the tip wrt the contour
        """
        max_index = np.max(roi_indices)+1
        max_dot_product = 0
        tip_index = roi_indices[0]

        if prev_tip is not None:
            weights_to_tip = compute_weights_df(contour, prev_tip)

        # compute new tip position
        for roi_index in roi_indices:
            window_indices = self._get_window_indices(roi_index, max_index)
            dot = lambda x: np.dot(np.exp2(displacements[x]) * normals[x], direction)
            if prev_tip is not None:
                dot_product = statistics.mean([weights_to_tip.loc[i][0] * dot(i) for i in window_indices])
            else:
                dot_product = statistics.mean([dot(i) for i in window_indices])
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                tip_index = roi_index

        return tip_index


class TipCorrection:

    """ Uses DBSCAN to spot badly placed tips and correct their position according to the
    next or previous known valid tip. Also corrects their ROI by setting it to the membrane. """

    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    @classmethod
    def from_config(cls, config):
        eps = config["TIP"]["EPS"] / config["PIXEL_SIZE"]
        min_samples = config["TIP"]["SAMPLES"]
        return cls(eps=eps, min_samples=min_samples)

    def get_tips(self, tubes, step):
        """ formats tubes to be used by DBSCAN. """
        return np.array([tubes[i].tip for i in range(len(tubes) - step)])

    def get_labels(self, tips):
        """ uses DBSCAN to detect clusters amongst tips. """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = pd.Series(dbscan.fit_predict(tips))
        return labels

    def get_principal_label(self, labels):
        """ get the biggest cluster label. """
        return labels.value_counts().index[0]

    def get_anomaly_series(self, is_anomaly):
        """ computes a list of list. sublist sequences of anomalies. for example if there are
         anomalies at index (wrt tubes) 1, 2, 3 and 7, it would return [[1, 2, 3], [7]]"""
        anomaly_series = list()
        is_anomaly_serie = False
        for i in range(len(is_anomaly)):
            if is_anomaly[i] == 1 and is_anomaly_serie:
                anomaly_series[-1].append(i)
            elif is_anomaly[i] == 1 and not is_anomaly_serie:
                anomaly_series.append([i])
                is_anomaly_serie = True
            else:
                is_anomaly_serie = False
        return anomaly_series

    def fix_tube(self, tubes, anomaly_serie, correction_tip, membrane, cnt_displ, step, shape):
        """ corrects tip, roi and displacement of a serie of adjacent anomalies. """
        for anomaly_index in anomaly_serie:
            tube = tubes[anomaly_index]
            tube.tip_index = project_on_contour(tube.contour, correction_tip)
            tube.roi_indices, _ = membrane.get_contour_indices(tube.contour, tube.tip_index)
            # compute displacement & direction
            tube.displacements = cnt_displ.get_displacements(
                current_contour=tube.contour,
                next_contour=tubes[anomaly_index + step].contour,
                current_normals=tube.N,
                roi_indices=tube.roi_indices,
                shape=shape
            )
            tube.direction = cnt_displ.get_direction(tube.displacements, tube.N)
            correction_tip = tube.tip
            tubes[anomaly_index] = tube
        return tubes

    def correct_tubes(self, tubes, membrane, cnt_displ, step, shape):
        """
        Finds the anomalies in tip positions with DBSCAN,
        Corrects the tip position by using the closest previous valid tip,
        Sets the ROI on the corrected tip,
        Computes the displacement of the corrected tip and ROI
        :param tubes: detected tubes
        :param membrane: Membrane class instance
        :param cnt_displ: ContourDisplacement class instance
        :param step: step parameter
        :param shape: frame shape
        :return:
        """
        tips = self.get_tips(tubes, step)
        labels = self.get_labels(tips)
        p_label = self.get_principal_label(labels)
        is_anomaly = labels.apply(lambda label: 1 if label != p_label else 0)
        anomaly_series = self.get_anomaly_series(is_anomaly)

        for anomaly_serie in anomaly_series:
            correction_tip = tubes[anomaly_serie[0] - 1].tip
            # backward prop in case no first tip is available
            if anomaly_serie[0] == 0:
                correction_tip = tubes[anomaly_serie[-1] + 1].tip
                anomaly_serie = reversed(anomaly_serie)
            tubes = self.fix_tube(tubes, anomaly_serie, correction_tip, membrane, cnt_displ, step, shape)
        return tubes

class VertexDetection:

    """ Deprecated """

    def __init__(self, radius, window_size):
        self.radius = radius
        self.window_size = window_size

    @classmethod
    def from_config(cls, config):
        radius = config["VERTEX"]["RADIUS"]
        window_size = config["TIP"]["WINDOW_SIZE"]
        return cls(radius=radius, window_size=window_size)

    def _get_window_indices(self, window_center_index, max_index):
        half_window = int(self.window_size / 2)
        return [index % max_index for index in
                range(window_center_index - half_window, window_center_index + half_window + 1)]

    def _spatial_filter(self, contour, prev_vertex, roi_indices):
        possible_vertex_inds = list()
        for i, point in enumerate(contour):
            if i in roi_indices:
                if (prev_vertex is not None and np.linalg.norm(point - prev_vertex) <= self.radius) \
                        or prev_vertex is None:
                    possible_vertex_inds.append(i)
        return possible_vertex_inds

    def _select_max_curvature(self, curvatures, indices):
        max_curvature = 0
        vertex_index = 0
        for i in indices:
            window_indices = self._get_window_indices(i, len(curvatures))
            curvature = sum([curvatures[j] for j in window_indices])
            if curvature > max_curvature:
                max_curvature = curvature
                vertex_index = i
        return vertex_index

    def detect_vertex(self, contour, curvatures, prev_vertex, roi_indices):
        possible_vertex_inds = self._spatial_filter(contour, prev_vertex, roi_indices)
        vertex_index = self._select_max_curvature(curvatures, possible_vertex_inds)
        return vertex_index


class VertexTrajectory:

    """ Deprecated """

    def __init__(self, degree):
        self.degree = degree

    def _polynomial_smoothing(self, ts, xs):
        poly_coeff = np.polyfit(x=ts, y=xs, deg=self.degree)
        poly = np.poly1d(poly_coeff)
        xs_smoothed = [round(poly(t)) for t in ts]
        return xs_smoothed

    def _smooth_trajectory(self, vertices):
        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        ts = [i for i in range(len(vertices))]

        xs_smoothed = self._polynomial_smoothing(ts, xs)
        ys_smoothed = self._polynomial_smoothing(ts, ys)

        return np.array([(x, y) for x, y in zip(xs_smoothed, ys_smoothed)])

    def smooth_vertices(self, vertices, contours):
        adjusted = self._smooth_trajectory(vertices)
        return [project_on_contour(contour, vertex) for contour, vertex in zip(contours, adjusted)]
