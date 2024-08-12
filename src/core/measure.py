import cv2 as cv
import numpy as np
from shapely.geometry import LineString
from shapely.geometry.multipoint import MultiPoint
from skimage.draw import line

from src.core.contour import project_on_contour
from src.core.utils import get_mask


class Measure:

    """ class that provides measurement computations on the pollen tube. """
    def __init__(self, timestep, pixel_size):
        """
        :param timestep: number of s in between frames
        :param pixel_size: um for one pixel unit
        """
        self.timestep = timestep
        self.pixel_size = pixel_size

    @classmethod
    def from_config(cls, config):
        timestep = config["TIMESTEP"]
        pixel_size = config["PIXEL_SIZE"]
        return cls(timestep=timestep, pixel_size=pixel_size)

    @staticmethod
    def get_abs_curv_scale(length, n_points):
        """
        define a scale from 0 to length having n_points equally spaced points
        :param length: length of the membrane
        :param n_points: number of points for membrane measurements (can be arbitrary but it is
        better derived from the actual number of points in the membrane contour. Indeed if n_points
        is bigger then we will not have exact measures, if it too small we lose data)
        :return: array of n_points from 0 to length
        """
        half_length = round(length / 2)
        return np.linspace(-half_length, half_length, n_points)

    def get_um_scale(self, scale):
        """
        uses um per pix value to cast the arbitrary scale to physical units
        :param scale: created by get_abs_curv_scale
        :return: scaled x scale
        """
        return self.pixel_size * scale

    def get_time_scale(self, length):
        """
        define a time scale with `length` increments of 1 or timestep
        :param length: length of the measurements
        :return: scale of time
        """
        return [self.timestep * i for i in range(length)]

    @staticmethod
    def region_mean_intensity(frame, mask):
        """
        measure the mean frame intensity inside the given mask
        :param frame: frame with intensity values
        :param mask: mask of the measurement region
        :return: mean intensity
        """
        roi = np.where(mask == 255)
        intensities = frame[roi]
        return np.mean(intensities)

    def area_growth(self, contour, prev_contour, prev_membrane_contour, shape):
        """
        compute the difference of areas given two contours
        the difference is expressed in physical units (um2 / s)
        :param contour: contour at timestep t
        :param prev_contour: contour at timestep t - 1
        :param prev_membrane_contour: membrane contour at timestep t - 1
        :param shape: shape used to compute mask
        :return: area growth at time t
        """
        if (prev_contour is None) and (prev_membrane_contour is None):
            return 0
        # get membrane extreme points
        m_ext_1 = prev_membrane_contour[0]
        m_ext_2 = prev_membrane_contour[-1]

        # project on current contour
        ind_1 = project_on_contour(contour, m_ext_1)
        ind_2 = project_on_contour(contour, m_ext_2)

        # cut current contour on region of interest (discarding points from the tube)
        start, end = min(ind_1, ind_2), max(ind_1, ind_2)
        # depending on the begining point of the contour the cut is not the same
        contour_1 = contour[start:end]
        index_contour_2 = [(end + i) % len(contour) for i in
                           range((len(contour) - end + start + 1))]
        contour_2 = contour[index_contour_2]
        # we use the smallest one (meaning not the tube part)
        if cv.arcLength(contour_1, closed=False) > cv.arcLength(contour_2, closed=False):
            contour = contour_2
        else:
            contour = contour_1

        # compute contour diff
        cnt_mask = get_mask(contour, shape, "fill")
        prev_mask = get_mask(prev_contour, shape, "fill")
        diff = np.subtract(cnt_mask, prev_mask)

        # measure area
        area = np.sum(diff == 255)
        area = (area * (self.pixel_size ** 2)) / self.timestep

        return area

    def growth_angle(self, direction, reference):
        """
        computes the angle in degrees between the vector `direction` and `reference`
        :param direction: direction vector
        :param reference: reference vector
        :return: angle between direction and reference in degrees
        """
        if np.linalg.norm(direction) == 0.0:
            unit_direction = direction
        else:
            unit_direction = direction / np.linalg.norm(direction)
        if np.linalg.norm(reference) == 0.0:
            unit_ref = reference
        else:
            unit_ref = reference / np.linalg.norm(reference)
        # Previous calculation method
        # rad_angle = np.arccos(np.clip(np.dot(unit_direction, unit_ref), -1.0, 1.0))

        # Current method
        v0 = unit_ref - np.array([0, 0])
        v1 = unit_direction - np.array([0, 0])
        rad_angle = np.arctan2(
            np.linalg.det([v0, v1]),
            np.dot(v0, v1)
        )
        return np.degrees(rad_angle)

    @staticmethod
    def membrane_intensity_distribution(frame, contour, normals, membrane_indices,
                                        membrane_thickness, membrane_xs, desired_xs):
        """
        spans normal vectors of the membrane backward to measure intensity along the normal direction
        for each point of the membrane contour.
        intensities are then averaged for each membrane contour point.
        additionaly, measures are interpolated to lie on the given scale (so that we can compare
        measures from frame to frame).
        :param membrane_indices: indices of the membrane wrt tube contour
        :param frame: frame with intensity values
        :param contour: tube contour
        :param normals: tube normals
        :param membrane_thickness: thickness of the membrane
        :param membrane_xs: curvilinear scale of the membrane
        :param desired_xs: curvilinear scale of where to make measures
        :return: intensity distribution over the membrane
        """
        intensities = []

        for ind in membrane_indices:
            mpoint = contour[ind]
            mnormal = normals[ind]
            end = np.round(mpoint - membrane_thickness * mnormal).astype(int)
            measurement_line = line(mpoint[0], mpoint[1], end[0], end[1])
            line_intensities = [frame[y, x] for x, y
                                in zip(measurement_line[0], measurement_line[1])
                                if x < frame.shape[1] and y < frame.shape[0]]
            intensity = sum(line_intensities) / len(line_intensities) if line_intensities else 0
            intensities.append(intensity)

        membrane_xs = np.array(membrane_xs)
        intensities = np.array(intensities)
        intensities = np.interp(desired_xs, membrane_xs, intensities)
        return intensities

    @staticmethod
    def membrane_curvatures(curvatures, membrane_indices, membrane_xs, desired_xs):
        """
        computes membrane curvatures adjusted on the xs scale (for comparison purposes).
        :param membrane_indices: indices of the membrane wrt tube contour
        :param curvatures: tube curvatures
        :param desired_xs: target curvilinear abscissa
        :param membrane_xs: curvilinear scale of the membrane
        :return: curvatures of the membrane at the given curvilinear abscissa xs
        """
        curvs = curvatures[membrane_indices]

        membrane_xs = np.array(membrane_xs)
        curvs = np.interp(desired_xs, membrane_xs, curvs)
        return curvs

    def tip_size(self, membrane_contour):
        """
        computes the size of the tip part of the pollen tube
        :param membrane_contour: contour of the membrane
        :return: size of the tip part in micrometers
        """
        pix_distance = np.linalg.norm(membrane_contour[0] - membrane_contour[-1])
        pix_distance *= self.pixel_size
        return pix_distance

    def growth_from_direction_new(self, prev_tip, prev_growth_direction, contour, shape):
        """
        computes growth defined by the displacement along the main growth direction
        :param prev_tip: coordinates of the previous tip
        :param prev_growth_direction: growth direction vector of the previous frame
        :param contour: contour of the current frame
        :param shape: frame shape
        :return: growth indicator in um
        """
        if (prev_tip is None) and (prev_growth_direction is None):
            return 0

        contour_line = LineString([(contour[i, 0], contour[i, 1])
                                        for i in range(len(contour))])
        end = prev_tip + prev_growth_direction * (shape[0] * shape[1])   # end point to "infinity"
        tip_growth_line = LineString([(prev_tip[0], prev_tip[1]), (end[0], end[1])])

        growth = 0
        if tip_growth_line.intersects(contour_line):
            intersec = tip_growth_line.intersection(contour_line)
            if isinstance(intersec, MultiPoint):
                intersec = intersec[0]
            intersec = np.array([intersec.x, intersec.y])
            growth = np.linalg.norm(intersec - prev_tip)
            growth *= self.pixel_size
        return growth

    def growth_from_direction(self, tip, growth_direction, next_contour, shape, step):
        """
        computes growth defined by the displacement along the main growth direction
        :param tip: coordinates of the tip
        :param growth_direction: main growth direction vector
        :param next_contour: contour at time t + step
        :param shape: frame shape
        :param step: time step to evaluate growth (number of frames)
        :return: growth indicator in um/s
        """
        next_contour_line = LineString([(next_contour[i, 0], next_contour[i, 1])
                                        for i in range(len(next_contour))])
        end = tip + growth_direction * (shape[0] * shape[1])   # end point to "infinity"
        tip_growth_line = LineString([(tip[0], tip[1]), (end[0], end[1])])

        growth = 0
        if tip_growth_line.intersects(next_contour_line):
            intersec = tip_growth_line.intersection(next_contour_line)
            if isinstance(intersec, MultiPoint):
                intersec = intersec[0]
            intersec = np.array([intersec.x, intersec.y])
            growth = np.linalg.norm(intersec - tip)
            growth *= self.pixel_size
            growth /= step * self.timestep
        return growth

    def growth_from_tip(self, tip, prev_tip):
        """
        computes growth defined by the distance between tips of consecutive frames.
        :param tip: coordinates of the tip at time t
        :param prev_tip: coordinates of the tip on the previous frame
        :return: growth indicator in um
        """
        if prev_tip is None:
            return 0
        growth = np.linalg.norm(tip - prev_tip)
        growth *= self.pixel_size
        return growth
