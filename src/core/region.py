import cv2 as cv
import numpy as np

from shapely.geometry import LineString, LinearRing
from src.core.utils import get_mask, sample_cnt
from src.utils import get_contour_ring, keep_best_inter


class Region:
    """ Regions should implement a method to return its mask. """
    def get_mask(self, **kwargs):
        raise NotImplementedError


class Membrane(Region):
    def __init__(self, length, thickness):
        """
        :param length: length of the membrane arc
        :param thickness: thickness of the membrane
        """
        self.length = length
        self.thickness = thickness

    @classmethod
    def from_config(cls, config):
        length = int(config["MEMBRANE"]["LENGTH"] / config["PIXEL_SIZE"])
        thickness = config["MEMBRANE"]["THICKNESS"]
        return cls(length=length, thickness=thickness)

    @classmethod
    def from_region_config(cls, config):
        """ this is a trick to define regions that have a different length from the membrane. """
        length = int(config["REGION"]["LENGTH"] / config["PIXEL_SIZE"])
        thickness = config["MEMBRANE"]["THICKNESS"]
        return cls(length=length, thickness=thickness)

    def get_contour_indices(self, contour, tip_index):
        """
        from a point of interest in the contour, expand a curve to its neighboring points wrt the
        contour with parameter length
        :param contour: array of contour points for the pollen tube
        :param tip_index: index of the point of the contour to expand the membrane froms
        :return: array of contour points for the membrane
        """
        curr_length, it = 0, 1
        membrane_cnt, membrane_indices, xs = [contour[tip_index]], [tip_index], [0]
        bot_cnt = [contour[tip_index]]
        top_cnt = [contour[tip_index]]
        max_index = len(contour)

        # while length hasn't reach the membrane length defined by the user
        while curr_length < self.length:
            # expand the indices to the left and right from the starting point
            bottom_index = (tip_index - it) % max_index
            top_index = (tip_index + it) % max_index
            membrane_cnt = [contour[bottom_index]] + membrane_cnt + [contour[top_index]]
            membrane_indices = [bottom_index] + membrane_indices + [top_index]
            bot_cnt.append(contour[bottom_index])
            top_cnt.append(contour[top_index])
            xs = [-cv.arcLength(np.array(bot_cnt), False)] + xs + [cv.arcLength(np.array(top_cnt), False)]
            curr_length = cv.arcLength(np.array(membrane_cnt), False)
            it += 1

        return membrane_indices, xs

    def get_mask(self, tube_contour, membrane_contour, shape):
        """
        builds a mask of the membrane by expanding a line around the membrane contour and
        restricting it to the intersection of the pollen tube contour
        :param tube_contour: array of contour points for the pollen tube
        :param membrane_contour: array of contour points for the membrane
        :param shape: shape of the produced mask (matches frame shape)
        :return: membrane mask
        """
        tube_mask = get_mask(tube_contour, shape, "fill")
        membrane_mask = get_mask(
            membrane_contour, shape, mask_type="lines", isClosed=False, thickness=self.thickness*2
        )
        mask = cv.bitwise_and(membrane_mask, tube_mask)
        return mask


class RegionA(Region):

    """ defines a region of type A """

    def __init__(self, depth):
        self.depth = depth

    @classmethod
    def from_config(cls, config):
        depth = round(config["REGION"]['A']["DEPTH"] / config["PIXEL_SIZE"])
        return cls(depth=depth)

    def get_mask(self, membrane_mask, membrane_contour, membrane_thickness, contour, shape):
        """
        computes the region A mask
        :param membrane_mask: binary mask of the membrane
        :param membrane_contour: array of contour points for the membrane
        :param membrane_thickness: thickness of the membrane (for compatibility)
        :param contour: contour of the pollen tube (for compatibility)
        :param shape: shape of the produced mask (matches frame shape)
        :return: region type A mask
        """
        start_cnt, end_cnt = sample_cnt(membrane_contour)
        imp = (start_cnt + end_cnt) / 2
        mmp = membrane_contour[len(membrane_contour) // 2]
        vector = imp - mmp
        vector /= np.linalg.norm(vector)
        vector = (np.round(vector*self.depth) + mmp).astype(np.int32)

        # construct a line going from the two end of the roi membrane
        line = np.polyfit(x=[start_cnt[0], end_cnt[0]], y=[start_cnt[1], end_cnt[1]], deg=1)
        a = line[0]

        # compute the translation towards the inside of the pollen tube of the first line
        translated_b = vector[1] - (vector[0]*a)
        translated_line = LineString([(x, a*x + translated_b) for x in range(0, shape[1])])

        # create a linear ring object of the pollen tube's contour
        contour_ring = get_contour_ring(contour)

        # check if there are any intersection between the translated line and the contour
        if translated_line.intersects(contour_ring):
            intersect = translated_line.intersection(contour_ring)
            inter_1, inter_2 = keep_best_inter(intersect)
            # middle of the two intersection points is the apex of the roi D cone
            point = np.array(
                [
                    (inter_1[0] + inter_2[0])/2,
                    (inter_1[1] + inter_2[1])/2
                ]
            ).astype(np.int32)

        # add the point to the contour to create the cone-shape roi
        point = np.expand_dims(np.round(point).astype(np.int32), axis=0)
        membrane_contour = np.append(membrane_contour, point, axis=0)

        # construct the cone-shaped mask
        mask = get_mask(membrane_contour, shape, mask_type="fill")
        mask = np.subtract(mask, membrane_mask)
        _, mask = cv.threshold(mask, 2, 255, cv.THRESH_BINARY)

        return mask


class RegionB(Region):
    """ defines a region of type B (Horseshoe following the membrane) """

    def __init__(self, thickness):
        self.thickness = thickness

    @classmethod
    def from_config(cls, config):
        thickness = int(config["REGION"]['B']["THICKNESS"] / config["PIXEL_SIZE"])
        return cls(thickness=thickness)

    def get_mask(self, membrane_mask, membrane_contour, membrane_thickness, contour, shape):
        """
        from membrane specs, builds the type B region mask
        :param membrane_mask: binary mask of the membrane
        :param membrane_contour: array of contour points for the membrane
        :param membrane_thickness: width of the membrane
        :param contour: contour of the pollen tube (for compatibility)
        :param shape: shape of the produced mask (matches frame shape)
        :return: region type B mask
        """
        thickness = 2 * self.thickness + membrane_thickness
        mask = get_mask(
            membrane_contour, shape, mask_type="lines", isClosed=False, thickness=thickness
        )
        cytoplasm_mask = get_mask(membrane_contour, shape, "fill")
        mask = cv.bitwise_and(mask, cytoplasm_mask)
        mask = np.subtract(mask, membrane_mask)
        _, mask = cv.threshold(mask, 2, 255, cv.THRESH_BINARY)
        return mask


class RegionC(Region):

    """ defines a region of type C (filled inside membrane) """

    def __init__(self, depth):
        self.depth = depth

    @classmethod
    def from_config(cls, config):
        depth = round(config["REGION"]['C']["DEPTH"] / config["PIXEL_SIZE"])
        return cls(depth=depth)

    def get_mask(self, membrane_mask, membrane_contour, membrane_thickness, contour, shape):
        """
        from membrane specs, builds the type C region mask
        :param membrane_mask: binary mask of the membrane (for compatibility)
        :param membrane_contour: array of contour points for the membrane
        :param membrane_thickness: thickness of the membrane (for compatibility)
        :param contour: contour of the pollen tube
        :param shape: shape of the produced mask (matches frame shape)
        :return: region type C mask
        """
        start_cnt, end_cnt = sample_cnt(membrane_contour)
        imp = (start_cnt + end_cnt) / 2
        mmp = membrane_contour[len(membrane_contour) // 2]

        vector = imp - mmp
        vector /= np.linalg.norm(vector)
        vector = (np.round(vector*self.depth) + mmp).astype(np.int32)

        # construct a line going from the two end of the roi membrane
        line = np.polyfit(x=[start_cnt[0], end_cnt[0]], y=[start_cnt[1], end_cnt[1]], deg=1)
        a = line[0]

        # compute the translation towards the inside of the pollen tube of the first line
        translated_b = vector[1] - (vector[0]*a)
        translated_line = LineString([(x, a*x + translated_b) for x in range(0, shape[1])])

        # create a linear ring object of the pollen tube's contour
        contour_ring = get_contour_ring(contour)

        # check if there are any intersection between the translated line and the contour
        if translated_line.intersects(contour_ring):
            intersect = translated_line.intersection(contour_ring)
            inter_1, inter_2 = keep_best_inter(intersect)

            # find the indices of the closest points in the contour to the intersection points
            idx_1 = np.sum(np.abs(contour - inter_1), axis=1).argmin()
            idx_2 = np.sum(np.abs(contour - inter_2), axis=1).argmin()
            start, end = min(idx_1, idx_2), max(idx_1, idx_2)

            sub_cont = contour[start:end+1]
            # if the middle point of the membrane is not contained in the sub contour
            # reverse the start and end indices of the sub contour to include it
            if not any(np.equal(sub_cont, mmp).all(1)):
                sub_cont = np.concatenate([contour[end:], contour[:start]])

            mask = get_mask(sub_cont, shape, mask_type="fill")

        return mask


class RegionD(Region):

    """ defines a region of type D """

    def __init__(self, depth, radius):
        self.radius = radius
        self.depth = depth

    @classmethod
    def from_config(cls, config):
        depth = round(config["REGION"]['D']["DEPTH"] / config["PIXEL_SIZE"])
        radius = round(config["REGION"]['D']["RADIUS"] / config["PIXEL_SIZE"])
        return cls(depth=depth, radius=radius)

    def get_mask(self, membrane_mask, membrane_contour, membrane_thickness, contour, shape):
        """
        computes the region D mask
        :param membrane_mask: binary mask of the membrane (for compatibility)
        :param membrane_contour: array of contour points for the membrane
        :param membrane_thickness: thickness of the membrane (for compatibility)
        :param contour: contour of the pollen tube (for compatibility)
        :param shape: shape of the produced mask (matches frame shape)
        :return: region type D mask
        """
        start_cnt, end_cnt = sample_cnt(membrane_contour)
        imp = (start_cnt + end_cnt) / 2
        mmp = membrane_contour[len(membrane_contour) // 2]

        vector = imp - mmp
        vector /= np.linalg.norm(vector)
        vector = (np.round(vector*self.depth) + mmp).astype(np.int32)

        # construct a line going from the two end of the roi membrane
        line = np.polyfit(x=[start_cnt[0], end_cnt[0]], y=[start_cnt[1], end_cnt[1]], deg=1)
        a = line[0]

        # compute the translation towards the inside of the pollen tube of the first line
        translated_b = vector[1] - (vector[0]*a)
        translated_line = LineString([(x, a*x + translated_b) for x in range(0, shape[1])])

        # create a linear ring object of the pollen tube's contour
        contour_ring = get_contour_ring(contour)

        # check if there are any intersection between the translated line and the contour
        if translated_line.intersects(contour_ring):
            intersect = translated_line.intersection(contour_ring)
            inter_1, inter_2 = keep_best_inter(intersect)
            # middle of the two intersection points is the center of the roi D circle
            center = np.array(
                [
                    (inter_1[0] + inter_2[0])/2,
                    (inter_1[1] + inter_2[1])/2
                ]
            ).astype(np.int32)

        # construct the mask of the roi
        center = np.round(center).astype(np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        mask = cv.circle(mask, (center[0], center[1]), self.radius, 255, thickness=-1)
        return mask
