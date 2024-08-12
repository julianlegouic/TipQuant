import datetime
import os
import re
import skvideo.io

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from shapely.geometry import LinearRing
from shapely.validation import explain_validity


def makedirs(path, exist_ok=True):
    """
    makes a directory if it does not exist, else does nothing
    :param path: path of the directory
    """
    try:
        os.makedirs(path, exist_ok=exist_ok)
    except FileExistsError:
        # removing existing files
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))


def load_video(b_io, video_file):
    """
    loads video to the BytesIO object
    :param b_io:  BytesIO Object
    :param video_file: path to the video file
    :return: path to the video file
    """
    with open(video_file, "wb") as out:  # Open temporary file as bytes
        out.write(b_io.getbuffer())
    return video_file


@st.cache(show_spinner=False)
def get_frames(b_io, file_path):
    """
    read BytesIO object from a video file
    :param b_io: BytesIO object
    :param file_path: path to the video file
    :return: a list of frames
    """
    video_file = load_video(b_io, file_path)
    video_stream = cv.VideoCapture(video_file)
    if not video_stream.isOpened():
        print("Error opening video file")
    frames = []
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            break
        frames.append(frame)
    video_stream.release()
    # In that case we have most likely a tif (sequence of images)
    if len(frames) == 1:
        _, frames = cv.imreadmulti(video_file)
    return frames


def save_frames(frames, video_file):
    """
    saves a list a frames to a video
    :param frames: list of frames
    :param video_file: destination file
    """
    _, ext = os.path.splitext(video_file)
    if ext == ".mp4":
        writer = skvideo.io.FFmpegWriter(video_file, outputdict={"-c:v": "libx264",
                                                             "-pix_fmt": "yuv444p"})

        for frame in frames:
            if len(frame.shape) == 2:
                frame = np.expand_dims(frame, axis=-1)
            writer.writeFrame(frame)
        writer.close()
    else:
        h, w, _ = frames[0].shape

        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        video = cv.VideoWriter(video_file, fourcc, 30, (w, h), 1)

        for frame in frames:
            video.write(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        video.release()


def read_video(video_path):
    """
    reads a video and return a list of frames
    :param video_path: path to the video file
    :return a list of frames
    """
    video_stream = cv.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise OSError("Error while opening video file. \
            Please make sure you loaded the video correctly.")
    frames = []
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            break
        frames.append(frame)
    video_stream.release()
    # In that case we have most likely a tif (sequence of images)
    if len(frames) == 1:
        _, frames = cv.imreadmulti(video_path)
    return frames


def save_output(output, result_dir):
    """
    save the computed data
    :param output: output of the algorithm from `src.core.run.get_data`
    :param result_dir: directory where the results are saved
    :return:
    """
    output_frames, measure_data = output
    data, membrane_intensities, membrane_xs, membrane_curvs, normals, contours, displacements = measure_data

    save_frames(output_frames, os.path.join(result_dir, "video.avi"))
    data.to_csv(os.path.join(result_dir, "data.csv"), index=False)
    np.savetxt(os.path.join(result_dir, "membrane_intensities.csv"), membrane_intensities, delimiter=';')
    np.savetxt(os.path.join(result_dir, "membrane_xs.csv"), membrane_xs, delimiter=';')
    np.savetxt(os.path.join(result_dir, "membrane_curvs.csv"), membrane_curvs, delimiter=';')
    np.save(os.path.join(result_dir, "normals.npy"), normals)
    np.save(os.path.join(result_dir, "contours.npy"), contours)
    np.save(os.path.join(result_dir, "displacements.npy"), displacements)


def read_output(result_dir):
    data = pd.read_csv(os.path.join(result_dir, "data.csv"))
    membrane_intensities = np.loadtxt(os.path.join(result_dir, "membrane_intensities.csv"), delimiter=';')
    membrane_xs = np.loadtxt(os.path.join(result_dir, "membrane_xs.csv"), delimiter=';')
    membrane_curvs = np.loadtxt(os.path.join(result_dir, "membrane_curvs.csv"), delimiter=';')
    return data, membrane_intensities, membrane_xs, membrane_curvs

def clear_directory(dir_path):
    """Clear all the content of a directory."""
    for root, _, files in os.walk(dir_path):
        for file in files:
            os.remove(os.path.join(root, file))


def get_contour_ring(contour):
        """
        we use shapely to compute intersections so we transform the contour as an array to
        a shapely linear ring
        :param contour: contour points
        :return: shapely linear ring corresponding to the contour
        """
        contour_ring = LinearRing(contour)

        # correct the contour if not valid (aka self-intersections exist)
        while not contour_ring.is_valid:
            # find the point of self-intersection
            p = re.compile("\[(.*?)\]")
            wrong_point = p.findall(explain_validity(contour_ring))
            if len(wrong_point) > 0:
                # extract the point as an array if founded
                wrong_point = wrong_point[0]
                wrong_point = np.array(list(map(int, wrong_point.split(' '))))
                wrong_index = np.where((contour == wrong_point).all(1))[0]
                if len(wrong_index) > 0:
                    # remove the point(s) (if multiple) from the contour
                    contour = np.delete(contour, wrong_index, axis=0)
                    contour_ring = LinearRing(contour)

        return contour_ring


def keep_best_inter(intersect):
    """Keep the two main points in the result of intersection with two curves.
    Warning, the function makes the assumption that ONLY TWO intersection points
    should exist. Help in the case where more than two points are returned by the
    shapely method.

    Note that it will not work properly if more than 2 intersection
    points actually exist.

    :param intersect: result of the shapely intersection method
    """
    np_intersect = np.round(
        np.array(intersect.array_interface()["data"])
    ).reshape(-1, 2).astype(np.int32)
    inter_1, inter_2 = None, None
    for pt in np_intersect:
        if inter_1 is None:
            inter_1 = pt
        elif np.linalg.norm(inter_1 - pt) < 2:
            inter_1 = np.mean([inter_1, pt], axis=0)
        elif inter_2 is None:
            inter_2 = pt
        else:
            inter_2 = np.mean([inter_2, pt], axis=0)
    return np.round(inter_1).astype(np.int32), np.round(inter_2).astype(np.int32)


def compute_weights_df(contour, prev_tip):
    """Construct a dataframe indexed by the contour indices, containing the
    weights for all contour indices.

    :param contour: contour of the pollen tube
    :param prev_tip: boolean indicating if tip on previous frame existed
    """
    # compute distance to the tip for all contour indices
    dist_to_tip = [np.linalg.norm(cnt-prev_tip) for cnt in contour]

    # construct a dataframe with all the distance-to-tip-weights for all indices
    weights_to_tip = pd.DataFrame(
        index=np.arange(len(contour)),
        data=(dist_to_tip - np.max(dist_to_tip))/(np.min(dist_to_tip) - np.max(dist_to_tip))
    )
    # in case of indexes overlapping, remove duplicate indices
    weights_to_tip = weights_to_tip[~weights_to_tip.index.duplicated(keep="first")]

    return weights_to_tip


def get_temp_dir(temp_dir):
    subdirs = [x[0] for x in os.walk(temp_dir)][1:]

    if len(subdirs) == 0:
        return None

    first_ts = datetime.datetime.strptime(os.path.split(subdirs[0])[-1], "%Y-%m-%d_%H-%M-%S")
    directory = subdirs[0]
    for subdir in subdirs:
        timestamp_str = os.path.split(subdir)[-1]
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        if timestamp > first_ts:
            directory = subdir
            first_ts = timestamp
    return directory
