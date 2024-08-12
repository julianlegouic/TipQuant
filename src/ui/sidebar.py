import os

import streamlit as st

from src.core.run import get_contour_preview
from src.utils import read_video


def video_params_sidebar(config, default_config):
    """ makes the sidebar for video parameters. uses default_config parameters and config
    for users parameters. """
    st.sidebar.markdown("# Video Parameters")
    pixel_ratio = st.sidebar.number_input(
        "Pixel size (micrometers)",
        value=float(default_config["PIXEL_SIZE"]),
        format="%.3f"
    )
    config["PIXEL_SIZE"] = pixel_ratio

    timestep = st.sidebar.number_input(
        "Timestep (s)",
        value=float(default_config["TIMESTEP"]),
        format="%.3f"
    )
    config["TIMESTEP"] = timestep


def model_sidebar(config, default_config, TEMP_VIDEO_PATH):
    """ makes the sidebar for model parameters. uses default_config parameters and config
    for users parameters. """

    st.sidebar.markdown("# Contour Smoothing")
    st.sidebar.markdown("##### Number of knots")
    knots_ratio = st.sidebar.number_input(
        "number of contour points chosen as spline knots",
        min_value=0.,
        value=float(default_config["CONTOUR"]["SPLINE_KNOTS_RATIO"])
    )
    config["CONTOUR"]["SPLINE_KNOTS_RATIO"] = knots_ratio
    preview = st.sidebar.button(label="Preview contour", key="cntprev")
    if preview:
        frames = read_video(TEMP_VIDEO_PATH)
        frame = get_contour_preview(frames, config)
        st.sidebar.image(frame, width=325)

    st.sidebar.markdown("# Tip Detection")
    st.sidebar.markdown("##### Step")
    step = st.sidebar.slider(
        "number of future frames to consider for growth estimation",
        min_value=default_config["TIP"]["STEP_MIN"],
        max_value=default_config["TIP"]["STEP_MAX"],
        value=default_config["TIP"]["STEP"], step=1
    )
    config["TIP"]["STEP"] = step


def membrane_sidebar(config, default_config):
    """ makes the sidebar for membrane parameters. uses default_config parameters and config
    for users parameters. """
    st.sidebar.markdown("# Plasma Membrane")
    length = st.sidebar.slider(
        "Length (um)",
        min_value=default_config["MEMBRANE"]["LENGTH_MIN"],
        max_value=default_config["MEMBRANE"]["LENGTH_MAX"],
        value=default_config["MEMBRANE"]["LENGTH"], step=0.5
    )
    membrane_thickness = st.sidebar.slider(
        "Thickness (pixels)",
        min_value=default_config["MEMBRANE"]["THICKNESS_MIN"],
        max_value=default_config["MEMBRANE"]["THICKNESS_MAX"],
        value=default_config["MEMBRANE"]["THICKNESS"], step=1
    )
    config["MEMBRANE"]["LENGTH"] = length
    config["MEMBRANE"]["THICKNESS"] = membrane_thickness


def region_sidebar(config, default_config, img_dir):
    """ makes the sidebar for region parameters. uses default_config parameters and config
    for users parameters. """
    st.sidebar.markdown("# Cytoplasm Region")
    region_name = st.sidebar.selectbox("Type", ('A', 'B', 'C', 'D'), index=1)
    st.sidebar.image(os.path.join(img_dir, f"Region{region_name}.png"), width=325)

    if region_name == 'A' or region_name == 'B':
        region_length = st.sidebar.slider(
            "Length (um)",
            min_value=config["REGION"]["LENGTH_MIN"],
            max_value=config["REGION"]["LENGTH_MAX"],
            value=config["REGION"]["LENGTH"], step=0.5,
            key="region_length"
        )
        config["REGION"]["LENGTH"] = region_length
        if region_name == 'A':
            region_depth = st.sidebar.slider(
                "Depth (um)",
                min_value=default_config["REGION"]['A']["DEPTH_MIN"],
                max_value=default_config["REGION"]['A']["DEPTH_MAX"],
                value=default_config["REGION"]['A']["DEPTH"], step=0.1,
                key="region_a_depth"
            )
            config["REGION"]['A']["DEPTH"] = region_depth
        else:
            region_thickness = st.sidebar.slider(
                "Thickness (um)",
                min_value=default_config["REGION"][region_name]["THICKNESS_MIN"],
                max_value=default_config["REGION"][region_name]["THICKNESS_MAX"],
                value=default_config["REGION"][region_name]["THICKNESS"], step=0.1, key="region_b_width"
            )
            config["REGION"]['B']["THICKNESS"] = region_thickness

    if region_name == 'C':
        region_depth = st.sidebar.slider(
            "Depth (um)",
            min_value=default_config["REGION"]['C']["DEPTH_MIN"],
            max_value=default_config["REGION"]['C']["DEPTH_MAX"],
            value=default_config["REGION"]['C']["DEPTH"], step=0.5,
            key="region_a_depth"
        )
        config["REGION"]['C']["DEPTH"] = region_depth

    if region_name == 'D':
        region_depth = st.sidebar.slider(
            "Depth (um)",
            min_value=default_config["REGION"]['D']["DEPTH_MIN"],
            max_value=default_config["REGION"]['D']["DEPTH_MAX"],
            value=default_config["REGION"]['D']["DEPTH"], step=0.1,
            key="region_d_depth"
        )
        config["REGION"]['D']["DEPTH"] = region_depth
        region_radius = st.sidebar.slider(
            "Radius (um)",
            min_value=default_config["REGION"]['D']["RADIUS_MIN"],
            max_value=default_config["REGION"]['D']["RADIUS_MAX"],
            value=default_config["REGION"]['D']["RADIUS"], step=0.1,
            key="region_d_radius"
        )
        config["REGION"]['D']["RADIUS"] = region_radius

    return region_name


def advanced_sidebar(config, default_config, img_dir):
    """ makes the sidebar for advanced parameters. uses default_config parameters and config
    for users parameters. """
    st.sidebar.markdown("### Advanced Settings")

    st.sidebar.markdown("#### Contour detection")
    st.sidebar.markdown("##### Gamma value")
    gamma = st.sidebar.slider(
        "Gamma coefficient for gamma correction",
        min_value=default_config["CONTOUR"]["GAMMA_MIN"],
        max_value=default_config["CONTOUR"]["GAMMA_MAX"],
        value=default_config["CONTOUR"]["GAMMA"], step=0.1
    )
    config["CONTOUR"]["GAMMA"] = gamma
    st.sidebar.markdown("##### Keeping mask visible")
    keep_mask = st.sidebar.checkbox("Mask visible option")
    config["CONTOUR"]["MASK"] = keep_mask

    # st.sidebar.markdown("##### Kernel size")
    # kernel_size = st.sidebar.slider(
    #     "pixel size of the kernel used for frame preprocessing",
    #     min_value=default_config["CONTOUR"]["KERNEL_SIZE_MIN"],
    #     max_value=default_config["CONTOUR"]["KERNEL_SIZE_MAX"],
    #     value=default_config["CONTOUR"]["KERNEL_SIZE"], step=2
    # )
    # config["CONTOUR"]["KERNEL_SIZE"] = kernel_size

    # st.sidebar.markdown("##### Sigma")
    # sigma = st.sidebar.slider(
    #     "standard deviation of the gaussian blur",
    #     min_value=default_config["CONTOUR"]["SIGMA_MIN"],
    #     max_value=default_config["CONTOUR"]["SIGMA_MAX"],
    #     value=default_config["CONTOUR"]["SIGMA"], step=2
    # )
    # config["CONTOUR"]["SIGMA"] = sigma

    # st.sidebar.markdown("#### Contour smoothing")
    # st.sidebar.markdown("##### Degree")
    # degree = st.sidebar.slider(
    #     "degree of the spline polynomials",
    #     min_value=default_config["CONTOUR"]["SPLINE_DEGREE_MIN"],
    #     max_value=default_config["CONTOUR"]["SPLINE_DEGREE_MAX"],
    #     value=default_config["CONTOUR"]["SPLINE_DEGREE"], step=1
    # )
    # config["CONTOUR"]["SPLINE_DEGREE"] = degree

    st.sidebar.markdown("#### Tip Detection")
    st.sidebar.markdown("##### Window size")
    window = st.sidebar.slider(
        "pixel size of the sliding window used to find maximum growth",
        min_value=default_config["TIP"]["WINDOW_SIZE_MIN"],
        max_value=default_config["TIP"]["WINDOW_SIZE_MAX"],
        value=default_config["TIP"]["WINDOW_SIZE"], step=1
    )
    config["TIP"]["WINDOW_SIZE"] = window
    st.sidebar.image(os.path.join(img_dir, "window.png"), width=325)

    st.sidebar.markdown("#### Tip Correction")
    st.sidebar.markdown("The two parameters below translate the spatial density of tips."
                        "From one frame to another, tips do not move randomly. Therefore, given a"
                        "radius around a tip, we expect to find some previous and next tips inside."
                        "Taking into account this behavior allows us to detect wrongly placed tips.")
    st.sidebar.markdown("##### Epsilon")
    eps = st.sidebar.slider(
        "distance (um) in which we expect to find min_samples tips",
        min_value=default_config["TIP"]["EPS_MIN"],
        max_value=default_config["TIP"]["EPS_MAX"],
        value=default_config["TIP"]["EPS"], step=0.5
    )
    config["TIP"]["EPS"] = eps

    st.sidebar.markdown("##### Min samples")
    sample = st.sidebar.slider(
        "number of tips expected in the epsilon radius",
        min_value=default_config["TIP"]["SAMPLES_MIN"],
        max_value=default_config["TIP"]["SAMPLES_MAX"],
        value=default_config["TIP"]["SAMPLES"], step=1
    )
    config["TIP"]["SAMPLES"] = sample
