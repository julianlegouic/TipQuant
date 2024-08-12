import cv2 as cv
import numpy as np


def get_mask(contour, shape, mask_type, **kwargs):
    """
    from a contour returns a binary mask
    :param contour: contour points as array of size Npoints x 2
    :param shape: shape of the image (required to have a mask of the same shape)
    :param mask_type: fill if an area is needed else lines to have a polyline draw
    :param kwargs: additional kwargs for the polylines function
    :return: mask of the contour
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if mask_type == "fill":
        cv.fillPoly(mask, pts=[contour], color=255)
    elif mask_type == "lines":
        cv.polylines(mask, [contour], color=255, **kwargs)
    else:
        raise NotImplementedError("Pick between 'fill' and 'lines' for mask_type parameter")
    return mask


def adjust_gamma(frame, gamma=1.0):
	"""
    build a lookup table mapping the pixel values [0, 255] to
	their adjusted gamma values
    :param frame: original frame
    :param gamma: gamma coefficient
    """
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype(np.uint8)
    # apply gamma correction using the lookup table
	return cv.LUT(frame, table)


def cast_to_gray(frame):
    """ cast a BGR frame to gray """
    if len(frame.shape) == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame


def cast_to_color(frame):
    """ cast a gray frame to RGB """
    if len(frame.shape) < 3:
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    return frame


def normalize_frames(frames):
    """ normalize frames to ensure max value is always 255 """
    stacked_frames = np.stack(frames, axis=0)
    max_value = np.max(stacked_frames)
    if max_value < 255:
        stacked_frames = stacked_frames / max_value
        stacked_frames *= 255
        frames = list(stacked_frames.astype(np.uint8))
    return frames

def sample_cnt(cnt):
    """Sample two points from contour which have not the same X or Y coordinates. """
    start, end = 0, -1
    alternate = 0
    p1, p2 = cnt[start], cnt[end]
    while ((p1[0] == p2[0]) or (p1[1] == p2[1])) and (start + abs(end) < cnt.shape[0]):
        if alternate % 2 == 0:
            start += 1
            alternate += 1
        else:
            end -= 1
            alternate -= 1
        p1, p2 = cnt[start], cnt[end]
    return p1, p2

def make_frame(frame, contour, membrane_mask, region_mask, tip):
    """
    from an original frame, adds useful information
    :param frame: original frame
    :param contour: contour points as array of size Npoints x 2
    :param membrane_mask: mask of the membrane
    :param region_mask: mask of the cytoplasm
    :param tip: tip location as array of 2 x 1
    :return: modified frame
    """
    frame = cast_to_color(frame)
    # draw membrane
    membrane_mask = cv.cvtColor(membrane_mask, cv.COLOR_GRAY2RGB)
    membrane_mask[np.where((membrane_mask == [255, 255, 255]).all(axis=2))] = (0, 255, 0)
    frame = cv.addWeighted(frame, 1, membrane_mask, 0.7, 0)
    # draw region
    region_mask = cv.cvtColor(region_mask, cv.COLOR_GRAY2RGB)
    region_mask[np.where((region_mask == [255, 255, 255]).all(axis=2))] = (0, 125, 0)
    frame = cv.addWeighted(frame, 1, region_mask, 0.5, 0)
    # draw tip
    cv.circle(frame, (tip[0], tip[1]), 3, (255, 0, 0), 2)
    # draw contour
    cv.drawContours(frame, [contour], -1, (255, 255, 255))
    return frame
