import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import splev

from src.core.contour import (ContourDetection, ContourParameterization,
                                  ContourCharacterization, ContourDisplacement, ContourROI)
from src.core.measure import Measure
from src.core.model import PollenTube
from src.core.region import Membrane, RegionA, RegionB, RegionC, RegionD
from src.core.tip import TipDetection, TipCorrection
from src.core.utils import cast_to_gray, make_frame, normalize_frames


def get_contour_preview(frames, config):

    """ Computes a preview image of the detected contour on a frame and its spline knots. """

    gray_frames = [frame.copy() for frame in frames]
    gray_frames = [cast_to_gray(frame) for frame in gray_frames]
    gray_frames = normalize_frames(gray_frames)

    # compute contour
    cnt_detect = ContourDetection.from_config(config)
    cnt_detect.get_threshold(gray_frames)
    cnt_param = ContourParameterization.from_config(config)
    contour = cnt_detect.detect_contour(frame=gray_frames[0], prev_contour=None)
    contour, tck, _ = cnt_param.parameterize_contour(contour, gray_frames[0].shape)
    # show knots
    cv.drawContours(frames[0], [contour], -1, (255, 255, 255), 1)
    xs_new, ys_new = splev(tck[0], tck, der=0)
    for x, y, u in zip(xs_new, ys_new, tck[0]):
        x, y = round(x), round(y)
        cv.circle(frames[0], (x, y), 1, (0, 0, 255), 1)
    non_zero = cv.findNonZero(gray_frames[0])
    # crop image
    xmin, xmax = non_zero[:, 0, 0].min(), non_zero[:, 0, 0].max()
    ymin, ymax = non_zero[:, 0, 1].min(), non_zero[:, 0, 1].max()
    frame = frames[0][ymin:ymax, xmin:xmax]
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


def get_tubes(frames, config, progress_bar=None):

    """ Process tubes (contour, normals, tip, ...) for each frame. """

    # read config
    step = config["TIP"]["STEP"]
    assert (config["TIMESTEP"] != 0 and config["PIXEL_SIZE"] != 0), \
        "Units are required. Please indicate pixel size and timestep."

    # ensure frames are grayscale
    gray_frames = [frame.copy() for frame in frames]
    gray_frames = [cast_to_gray(frame) for frame in gray_frames]
    shape = gray_frames[0].shape
    gray_frames = normalize_frames(gray_frames)

    # initialize modules with parameters
    cnt_roi = ContourROI()
    cnt_detect = ContourDetection.from_config(config)
    cnt_param = ContourParameterization.from_config(config)
    cnt_charac = ContourCharacterization()
    cnt_displ = ContourDisplacement()
    tip_detect = TipDetection.from_config(config)
    tip_correct = TipCorrection.from_config(config)

    # detect contours
    if progress_bar is not None:
        progress_bar("Detect contour", len(gray_frames))

    tubes = list()
    prev_contour = None
    cnt_detect.get_threshold(gray_frames)

    for frame in tqdm(gray_frames, desc="Detecting Contour", total=len(gray_frames)):
        tube = PollenTube()
        # detect contour
        contour = cnt_detect.detect_contour(frame=frame, prev_contour=prev_contour)
        # parameterize
        tube.contour, tube.tck, tube.us = cnt_param.parameterize_contour(contour, shape)
        prev_contour = tube.contour
        # characterize
        tube.T, tube.N, tube.curvs = cnt_charac.characterize_contour(tube.tck, tube.us)
        tubes.append(tube)

        if progress_bar is not None:
            progress_bar.update()

    if progress_bar is not None:
        progress_bar("Detect TIP", len(tubes) - step)

    for i in tqdm(range(len(tubes) - step), desc="Detecting TIP", total=(len(tubes) - step)):
        # find roi
        tubes[i].roi_indices = cnt_roi.get_roi(
            current_contour=tubes[i].contour,
            next_contour=tubes[i + step].contour,
            shape=shape
        )
        # detect tip
        tubes[i].displacements = cnt_displ.get_displacements(
            current_contour=tubes[i].contour,
            next_contour=tubes[i + step].contour,
            current_normals=tubes[i].N,
            roi_indices=tubes[i].roi_indices,
            shape=shape
        )
        tubes[i].direction = cnt_displ.get_direction(
            displacements=tubes[i].displacements,
            normals=tubes[i].N
        )
        tubes[i].tip_index = tip_detect.get_tips_from_normals(
            normals=tubes[i].N,
            displacements=tubes[i].displacements,
            direction=tubes[i].direction,
            roi_indices=tubes[i].roi_indices,
            prev_tip=tubes[i-1].tip if i > 0 else None,
            contour=tubes[i].contour
        )

        if progress_bar is not None:
            progress_bar.update()

    membrane = Membrane.from_config(config)
    tubes = tip_correct.correct_tubes(tubes, membrane, cnt_displ, step, shape)
    return tubes


def get_data(frames, tubes, config, region_name, progress_bar=None):

    """ Computes measurement data from the tubes. """

    # read config
    step = config["TIP"]["STEP"]

    # ensure frames are grayscale
    gray_frames = [frame.copy() for frame in frames]
    gray_frames = [cast_to_gray(frame) for frame in gray_frames]
    shape = gray_frames[0].shape

    # initialize modules
    membrane = Membrane.from_config(config)
    membrane_for_region = Membrane.from_region_config(config)
    if region_name == 'A':
        region = RegionA.from_config(config)
    elif region_name == 'B':
        region = RegionB.from_config(config)
    elif region_name == 'C':
        region = RegionC.from_config(config)
    else:
        region = RegionD.from_config(config)
    measure = Measure.from_config(config)

    # get scales
    time_scale = measure.get_time_scale(len([tube for tube in tubes if tube.tip_index is not None]))
    # for xs we need to know approximately the number of points in the membrane
    n_xs_point = len(membrane.get_contour_indices(tubes[0].contour, tubes[0].tip_index)[0])
    xs_scale = measure.get_abs_curv_scale(length=membrane.length, n_points=n_xs_point)
    um_scale = measure.get_um_scale(xs_scale)

    # measures : we can store most data in tabular form but for the membrane we have to store
    # distributions so we will keep a separate array (pandas does handle well arrays in cells).
    output_frames = []
    data = pd.DataFrame(columns=["membrane_intensity_mean", "cytoplasm_intensity_mean",
                                 "area_growth", "tip_size",
                                 "growth_vec_x", "growth_vec_y", "growth_from_direction",
                                 "growth_from_tip", "growth_direction_angle"])
    membranes_intensities = list()
    membranes_curvatures = list()
    normals = list()
    contours = list()
    displacements = list()

    if progress_bar is not None:
        progress_bar("Data Measure", len(frames) - step)

    for i in tqdm(range(len(frames) - step), desc="Data Measure", total=(len(frames) - step)):

        tube = tubes[i]

        # membrane mask
        tube.membrane_indices, membrane_xs = membrane.get_contour_indices(tube.contour, tube.tip_index)
        membrane_mask = membrane.get_mask(tube.contour, tube.membrane_contour, shape)

        # for region masks as it can be of different length we use the membrane for region object
        mr_indices, _ = membrane_for_region.get_contour_indices(tube.contour, tube.tip_index)
        membrane_region_cnt = tube.contour[mr_indices]
        mr_mask = membrane_for_region.get_mask(tube.contour, membrane_region_cnt, shape)
        region_mask = region.get_mask(mr_mask, membrane_region_cnt, membrane_for_region.thickness, tube.contour, shape)

        normals.append(tube.N)
        contours.append(tube.contour)
        displacements.append(tube.displacements)

        # Area growth
        area_growth = measure.area_growth(
            contour=tube.contour,
            prev_contour=tubes[i - 1].contour if i > 0 else None,
            prev_membrane_contour=tubes[i - 1].contour[tubes[i - 1].membrane_indices] if i > 0 else None,
            shape=shape
        )
        data.loc[i, "area_growth"] = area_growth

        # Growth direction angle
        growth_angle = measure.growth_angle(
            direction=tube.direction,
            reference=tubes[0].direction
        )
        data.loc[i, "growth_direction_angle"] = growth_angle

        # Cytoplasm intensity
        cyto_intensity = measure.region_mean_intensity(
            frame=gray_frames[i],
            mask=region_mask
        )
        data.loc[i, "cytoplasm_intensity_mean"] = cyto_intensity

        # Membrane intensity
        memb_intensity = measure.region_mean_intensity(
            frame=gray_frames[i],
            mask=membrane_mask
        )
        data.loc[i, "membrane_intensity_mean"] = memb_intensity

        # Membrane intensity distribution
        membrane_distrib = measure.membrane_intensity_distribution(
            frame=gray_frames[i],
            contour=tube.contour,
            normals=tube.N,
            membrane_indices=tube.membrane_indices,
            membrane_thickness=membrane.thickness,
            membrane_xs=membrane_xs,
            desired_xs=xs_scale
        )
        membranes_intensities.append(membrane_distrib)

        # Membrane curvatures distribution
        current_curvs = measure.membrane_curvatures(
            curvatures=tube.curvs,
            membrane_indices=tube.membrane_indices,
            membrane_xs=membrane_xs,
            desired_xs=xs_scale
        )
        membranes_curvatures.append(current_curvs)

        # Tip size
        tip_size = measure.tip_size(tube.membrane_contour)
        data.loc[i, "tip_size"] = tip_size

        # Growth direction
        data.loc[i, "growth_vec_x"] = tube.direction[0]
        data.loc[i, "growth_vec_y"] = tube.direction[1]

        # Growth from direction
        growth_from_dir = measure.growth_from_direction(
            tip=tube.tip,
            growth_direction=tube.direction,
            next_contour=tubes[i + step].contour,
            shape=shape,
            step=step
        )
        data.loc[i, "growth_from_direction"] = growth_from_dir

        # Growth from tip
        growth_from_tip = measure.growth_from_tip(
            tip=tube.tip,
            prev_tip=tubes[i - 1].tip if i > 0 else None,
        )
        data.loc[i, "growth_from_tip"] = growth_from_tip

        # Tip location
        data.loc[i, "tip_x"] = tube.tip[0]
        data.loc[i, "tip_y"] = tube.tip[1]

        # Draw frame
        output_frame = make_frame(
            frame=frames[i],
            contour=tube.contour,
            membrane_mask=membrane_mask,
            region_mask=region_mask,
            tip=tube.tip
        )
        output_frames.append(output_frame)

        if progress_bar is not None:
            progress_bar.update()

    data["time"] = time_scale
    return output_frames, (data, np.array(membranes_intensities), um_scale,
                           np.array(membranes_curvatures), np.array(normals, dtype="object"),
                           np.array(contours, dtype="object"), np.array(displacements, dtype="object"))
