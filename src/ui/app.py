import datetime
import os
import platform
import toml
import streamlit as st

from src.core.run import get_tubes, get_data
from src.ui.plot import (plot_cytoplasm_intensity, plot_growth_area, plot_membrane_heatmap,
                         plot_membrane_intensity, plot_direction_angle)
from src.ui.progressionbar import ProgressionBar
from src.ui.sidebar import (video_params_sidebar, model_sidebar, membrane_sidebar,
                            region_sidebar, advanced_sidebar)
from src.ui.utils import write_config, local_css
from src.utils import clear_directory, makedirs, save_frames, get_frames, save_output, read_video, read_output

TMP_CODEC = "mp4v" if platform.system() == "Windows" else "H264"
RESULT_CODEC = "H264"
# tipQuant directory
MAIN_DIRECTORY = '/'.join(os.path.realpath(__file__).split('/')[:-3])
CONFIG_PATH = os.path.join(MAIN_DIRECTORY, "src/config.toml")
DEFAULT_CONFIG = toml.load(CONFIG_PATH)
IMG_PATH = os.path.join(MAIN_DIRECTORY, "data/assets")
TMP_RAW_DIR = os.path.join(MAIN_DIRECTORY, "tmp/raw")
TMP_MASK_DIR = os.path.join(MAIN_DIRECTORY, "tmp/mask")
TMP_RAW_VIDEO_PATH = os.path.join(MAIN_DIRECTORY, "tmp/raw/tmp_video_raw.mp4")
TMP_MASK_VIDEO_PATH = os.path.join(MAIN_DIRECTORY, "tmp/mask/tmp_video_mask.mp4")
TMP_RAW_OUTPUT_PATH = os.path.join(MAIN_DIRECTORY, "tmp/raw/tmp_video_raw_output.mp4")
TMP_MASK_OUTPUT_PATH = os.path.join(MAIN_DIRECTORY, "tmp/mask/tmp_video_mask_output.mp4")
config = DEFAULT_CONFIG.copy()


def main():
    st.set_page_config(page_title="TipQUANT", layout="wide")
    st.title("TipQUANT")
    local_css("./src/css/style.css")

    # Videos
    col_vid_raw, col_vid_mask = st.columns(2)
    with col_vid_raw:
        st.header("Primary video")
        with st.container(height=80, border=False):
            st.caption("Used to analyze a single video, or as the reference for contour detection when analyzing two-channel recordings of the same cell. Only one file should be uploaded.")
        uploaded_file_raw = st.file_uploader("Select file", type=["mp4", "avi", "tif", "m4v"], key="uf_raw")
    with col_vid_mask:
        st.header("Second parallel video (optional)")
        with st.container(height=80, border=False):
            st.caption("Upload the second channel's video from the same cell for two-channel analysis. Measurements are based on contours detected in the primary video. Only one file should be uploaded.")
        uploaded_file_mask = st.file_uploader("Select file", type=["mp4", "avi", "tif", "m4v"], key="uf_mask")

    load_video = st.button("Load video(s)", use_container_width=True)
    video_slot = st.empty()
    run = st.button("Run", use_container_width=True)

    # Sidebar
    st.sidebar.markdown("# Output")
    directory = st.sidebar.text_input("Output directory", "output")
    video_params_sidebar(config, DEFAULT_CONFIG)
    model_sidebar(config, DEFAULT_CONFIG, TMP_RAW_VIDEO_PATH)
    membrane_sidebar(config, DEFAULT_CONFIG)
    region_name = region_sidebar(config, DEFAULT_CONFIG, IMG_PATH)
    advanced_sidebar(config, DEFAULT_CONFIG, IMG_PATH)

    # Action
    if load_video:
        # Clear both temporary directories to avoid any data corruption
        clear_directory(TMP_RAW_DIR)
        clear_directory(TMP_MASK_DIR)
        # Currently we don't have access to the uploaded file path so we have to save the video
        # in a temporary file which we know the localisation in order to load it properly
        # https://github.com/streamlit/streamlit/issues/896.
        if uploaded_file_raw:
            with st.spinner("Loading video ..."):
                frames = get_frames(uploaded_file_raw, TMP_RAW_VIDEO_PATH)
                save_frames(frames, TMP_RAW_VIDEO_PATH, force_codec=TMP_CODEC)
        if uploaded_file_mask:
            with st.spinner("Loading video ..."):
                frames = get_frames(uploaded_file_mask, TMP_MASK_VIDEO_PATH)
                save_frames(frames, TMP_MASK_VIDEO_PATH, force_codec=TMP_CODEC)

    if os.path.exists(TMP_RAW_VIDEO_PATH):
        video_slot.video(TMP_RAW_VIDEO_PATH)

    if run:
        with st.spinner("Running ..."):
            progress_bar = ProgressionBar()
            raw_frames = read_video(TMP_RAW_VIDEO_PATH)
            assert len(raw_frames) > 0, "The video should contain at least one frame"
            mask_frames = None
            if uploaded_file_mask and os.path.exists(TMP_MASK_VIDEO_PATH):
                mask_frames = read_video(TMP_MASK_VIDEO_PATH)
                assert len(raw_frames) == len(mask_frames), "Raw video and mask video should have the same number of frames"

            # compute
            tubes = get_tubes(raw_frames, config, progress_bar)

            output_raw = get_data(raw_frames, tubes, config, region_name, progress_bar, "raw")
            output_frames_raw, measure_data_raw = output_raw
            data_raw, membrane_intensities_raw, membrane_xs_raw, _, _, _, _ = measure_data_raw
            save_frames(output_frames_raw, TMP_RAW_OUTPUT_PATH, force_codec=RESULT_CODEC)

            if mask_frames is not None:
                output_mask = get_data(mask_frames, tubes, config, region_name, progress_bar, "mask")
                output_frames_mask, measure_data_mask = output_mask
                data_mask, membrane_intensities_mask, membrane_xs_mask, _, _, _, _ = measure_data_mask
                save_frames(output_frames_mask, TMP_MASK_OUTPUT_PATH, force_codec=RESULT_CODEC)

        with st.spinner("Saving results..."):
            # save into target directory
            if uploaded_file_raw is None:
                raise AttributeError("Raw video is required")
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            raw_video_name, _ = os.path.splitext(uploaded_file_raw.name)
            result_dir_raw = os.path.join("data", directory, f"{time}_{raw_video_name}")
            makedirs(result_dir_raw)
            save_output(output_raw, result_dir_raw)
            write_config(config, os.path.join(result_dir_raw, "config.toml"))
            heatmap = plot_membrane_heatmap(data_raw, membrane_intensities_raw, membrane_xs_raw, "Jet", None)
            heatmap.write_html(os.path.join(result_dir_raw, "heatmap.html"))
            heatmap.write_image(os.path.join(result_dir_raw, "heatmap.svg"))
            # save into temp directory
            save_output(output_raw, TMP_RAW_DIR)

            # repeat for mask video if exists
            if mask_frames is not None and uploaded_file_mask is not None:
                mask_video_name, _ = os.path.splitext(uploaded_file_mask.name)
                result_dir_mask = os.path.join("data", directory, f"{time}_{mask_video_name}")
                makedirs(result_dir_mask)
                save_output(output_mask, result_dir_mask)
                write_config(config, os.path.join(result_dir_mask, "config.toml"))
                heatmap = plot_membrane_heatmap(data_mask, membrane_intensities_mask, membrane_xs_mask, "Jet", None)
                heatmap.write_html(os.path.join(result_dir_mask, "heatmap.html"))
                heatmap.write_image(os.path.join(result_dir_mask, "heatmap.svg"))
                save_output(output_mask, TMP_MASK_DIR)

        progress_bar.success("Measures are completed")

    raw_filelist = [f for f in os.listdir(TMP_RAW_DIR)]
    mask_filelist = []
    mask_filelist = [f for f in os.listdir(TMP_MASK_DIR)]

    if len(raw_filelist) > 2:
        # show video
        video_slot.video(TMP_RAW_OUTPUT_PATH)
        data_raw, membrane_intensities_raw, membrane_xs_raw, _ = read_output(TMP_RAW_DIR)
        data_mask, membrane_intensities_mask, membrane_xs_mask = None, None, None
        if uploaded_file_mask and len(mask_filelist) > 2:
            data_mask, membrane_intensities_mask, membrane_xs_mask, _ = read_output(TMP_MASK_DIR)

        # make plots
        st.header("Measures")

        window_size = st.slider("Bandwidth for smoothing curves with a moving average",
                                min_value=1,
                                max_value=10,
                                value=1,
                                step=1)

        aggregation_seconds = st.slider("Number of seconds to aggregate area growth",
                                min_value=0,
                                max_value=15,
                                value=0,
                                step=5)

        if aggregation_seconds == 0:
            aggregation_seconds = None

        growth_area_raw = plot_growth_area(data_raw, window_size, False, aggregation_seconds)
        st.plotly_chart(
            plot_growth_area(data_raw, window_size, True, aggregation_seconds),
            width="stretch"
        )
        direction_angle_raw = plot_direction_angle(data_raw, window_size, final=False)

        colorscale = st.selectbox("Color",
                                  ("Viridis", "Inferno", "Plasma", "Jet",
                                   "Oranges", "solar", "deep", "Purp"),
                                  index=3)
        smooth = st.selectbox("Smoothing", ("None", "Fast", "Best"), index=0)
        other_figs_label = st.selectbox("Select graph to overlay",
                                        options=("None",
                                                 "Membrane mean intensity",
                                                 "Cytoplasm mean intensity",
                                                 "Growth area",
                                                 "Direction angle"),
                                        )

        membrane_intensity_raw = plot_membrane_intensity(
            data_raw, window_size, False)
        cytoplasm_intensity_raw = plot_cytoplasm_intensity(
            data_raw, window_size, False)
        labels_to_plot = {
            "Membrane mean intensity": membrane_intensity_raw,
            "Cytoplasm mean intensity": cytoplasm_intensity_raw,
            "Growth area": growth_area_raw,
            "Direction angle": direction_angle_raw
        }
        other_fig = labels_to_plot[other_figs_label] if other_figs_label != "None" else None
        heatmap_raw = plot_membrane_heatmap(
            data_raw, membrane_intensities_raw, membrane_xs_raw, colorscale, other_fig, smooth.lower())

        if uploaded_file_mask and len(mask_filelist) > 2:
            membrane_intensity_mask = plot_membrane_intensity(
                data_mask, window_size, False)
            cytoplasm_intensity_mask = plot_cytoplasm_intensity(
                data_mask, window_size, False)
            labels_to_plot = {
                "Membrane mean intensity": membrane_intensity_mask,
                "Cytoplasm mean intensity": cytoplasm_intensity_mask,
                "Growth area": growth_area_raw,
                "Direction angle": direction_angle_raw
            }
            other_fig = labels_to_plot[other_figs_label] if other_figs_label != "None" else None
            heatmap_mask = plot_membrane_heatmap(
                data_mask, membrane_intensities_mask, membrane_xs_mask, colorscale, other_fig, smooth.lower())

            raw_data_column, mask_data_column = st.columns(2)
            with raw_data_column:
                st.subheader("Raw video measures")
                st.plotly_chart(
                    plot_membrane_intensity(data_raw, window_size, True),
                    width="stretch",
                )
                st.plotly_chart(
                    plot_cytoplasm_intensity(data_raw, window_size, True),
                    width="stretch"
                )
                st.plotly_chart(
                    heatmap_raw,
                    width="stretch"
                )
            with mask_data_column:
                st.subheader("Mask video measures")
                st.plotly_chart(
                    plot_membrane_intensity(data_mask, window_size, True),
                    width="stretch",
                )
                st.plotly_chart(
                    plot_cytoplasm_intensity(data_mask, window_size, True),
                    width="stretch"
                )
                st.plotly_chart(
                    heatmap_mask,
                    width="stretch"
                )
        else:
            st.subheader("Raw video measures")
            st.plotly_chart(
                plot_membrane_intensity(data_raw, window_size, True),
                width="stretch",
            )
            st.plotly_chart(
                plot_cytoplasm_intensity(data_raw, window_size, True),
                width="stretch"
            )
            st.plotly_chart(
                heatmap_raw,
                width="stretch"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(e)
