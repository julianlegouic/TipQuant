import datetime
import os
import platform
import toml
import streamlit as st

from src.core.run import get_tubes, get_data
from src.ui.plot import (plot_cytoplasm_intensity, plot_growth_area, plot_membrane_heatmap,
                         plot_membrane_intensity)
from src.ui.progressionbar import ProgressionBar
from src.ui.sidebar import (video_params_sidebar, model_sidebar, membrane_sidebar,
                            region_sidebar, advanced_sidebar)
from src.ui.utils import write_config, local_css
from src.utils import clear_directory, makedirs, save_frames, get_frames, save_output, read_video, read_output

TMP_CODEC = "mp4v" if platform.system() == "Windows" else "H264"
RESULT_CODEC = "H264"
# tipQuant directory
MAIN_DIRECTORY = '/'.join(os.path.realpath(__file__).split('/')[:-3])
TMP_CODEC = "mp4v" if platform.system() == "Windows" else "H264"
RESULT_CODEC = "H264"
# tipQuant directory
MAIN_DIRECTORY = '/'.join(os.path.realpath(__file__).split('/')[:-3])
CONFIG_PATH = os.path.join(MAIN_DIRECTORY, "src/config.toml")
DEFAULT_CONFIG = toml.load(CONFIG_PATH)
IMG_PATH = os.path.join(MAIN_DIRECTORY, "data/assets")
TMP_PRIMARY_DIR = os.path.join(MAIN_DIRECTORY, "tmp/primary")
TMP_SECONDARY_DIR = os.path.join(MAIN_DIRECTORY, "tmp/secondary")
TMP_PRIMARY_VIDEO_PATH = os.path.join(MAIN_DIRECTORY, "tmp/primary/tmp_video_primary.mp4")
TMP_SECONDARY_VIDEO_PATH = os.path.join(MAIN_DIRECTORY, "tmp/secondary/tmp_video_secondary.mp4")
TMP_PRIMARY_OUTPUT_PATH = os.path.join(MAIN_DIRECTORY, "tmp/primary/tmp_video_primary_output.mp4")
TMP_SECONDARY_OUTPUT_PATH = os.path.join(MAIN_DIRECTORY, "tmp/secondary/tmp_video_secondary_output.mp4")
config = DEFAULT_CONFIG.copy()


def main():
    st.set_page_config(page_title="TipQUANT", layout="wide")
    st.title("TipQUANT")
    local_css("./src/css/style.css")

    # Videos
    col_vid_primary, col_vid_secondary = st.columns(2)
    with col_vid_primary:
        st.header("Primary video")
        with st.container(height=80, border=False):
            st.caption("Used to analyze a single video, or as the reference for contour detection when analyzing two-channel recordings of the same cell. Only one file should be uploaded.")
        uploaded_file_primary = st.file_uploader("Select file", type=["mp4", "avi", "tif", "m4v"], key="uf_primary")
    with col_vid_secondary:
        st.header("Second parallel video (optional)")
        with st.container(height=80, border=False):
            st.caption("Upload the second channel's video from the same cell for two-channel analysis. Measurements are based on contours detected in the primary video. Only one file should be uploaded.")
        uploaded_file_secondary = st.file_uploader("Select file", type=["mp4", "avi", "tif", "m4v"], key="uf_secondary")

    load_video = st.button("Load video(s)", use_container_width=True)
    video_slot = st.empty()
    run = st.button("Run", use_container_width=True)

    # Sidebar
    st.sidebar.markdown("# Output")
    directory = st.sidebar.text_input("Output directory", "output")
    video_params_sidebar(config, DEFAULT_CONFIG)
    model_sidebar(config, DEFAULT_CONFIG, TMP_PRIMARY_VIDEO_PATH)
    membrane_sidebar(config, DEFAULT_CONFIG)
    region_name = region_sidebar(config, DEFAULT_CONFIG, IMG_PATH)
    advanced_sidebar(config, DEFAULT_CONFIG, IMG_PATH)

    # Action
    if load_video:
        # Clear both temporary directories to avoid any data corruption
        clear_directory(TMP_PRIMARY_DIR)
        clear_directory(TMP_SECONDARY_DIR)
        # Currently we don't have access to the uploaded file path so we have to save the video
        # in a temporary file which we know the localisation in order to load it properly
        # https://github.com/streamlit/streamlit/issues/896.
        if uploaded_file_primary:
            with st.spinner("Loading video ..."):
                frames = get_frames(uploaded_file_primary, TMP_PRIMARY_VIDEO_PATH)
                save_frames(frames, TMP_PRIMARY_VIDEO_PATH, force_codec=TMP_CODEC)
        if uploaded_file_secondary:
            with st.spinner("Loading video ..."):
                frames = get_frames(uploaded_file_secondary, TMP_SECONDARY_VIDEO_PATH)
                save_frames(frames, TMP_SECONDARY_VIDEO_PATH, force_codec=TMP_CODEC)

    if os.path.exists(TMP_PRIMARY_VIDEO_PATH):
        video_slot.video(TMP_PRIMARY_VIDEO_PATH)

    if run:
        with st.spinner("Running ..."):
            progress_bar = ProgressionBar()
            primary_frames = read_video(TMP_PRIMARY_VIDEO_PATH)
            assert len(primary_frames) > 0, "The video should contain at least one frame"
            secondary_frames = None
            if uploaded_file_secondary and os.path.exists(TMP_SECONDARY_VIDEO_PATH):
                secondary_frames = read_video(TMP_SECONDARY_VIDEO_PATH)
                assert len(primary_frames) == len(secondary_frames), "Primary video and secondary video should have the same number of frames"

            # compute
            tubes = get_tubes(primary_frames, config, progress_bar)

            output_primary = get_data(primary_frames, tubes, config, region_name, progress_bar, "primary")
            output_frames_primary, measure_data_primary = output_primary
            data_primary, membrane_intensities_primary, membrane_xs_primary, _, _, _, _ = measure_data_primary
            save_frames(output_frames_primary, TMP_PRIMARY_OUTPUT_PATH, force_codec=RESULT_CODEC)

            if secondary_frames is not None:
                output_secondary = get_data(secondary_frames, tubes, config, region_name, progress_bar, "secondary")
                output_frames_secondary, measure_data_secondary = output_secondary
                data_secondary, membrane_intensities_secondary, membrane_xs_secondary, _, _, _, _ = measure_data_secondary
                save_frames(output_frames_secondary, TMP_SECONDARY_OUTPUT_PATH, force_codec=RESULT_CODEC)

        with st.spinner("Saving results..."):
            # save into target directory
            if uploaded_file_primary is None:
                raise AttributeError("Primary video is required")
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            primary_video_name, _ = os.path.splitext(uploaded_file_primary.name)
            result_dir_primary = os.path.join("data", directory, f"{time}_{primary_video_name}")
            makedirs(result_dir_primary)
            save_output(output_primary, result_dir_primary)
            write_config(config, os.path.join(result_dir_primary, "config.toml"))
            heatmap = plot_membrane_heatmap(data_primary, membrane_intensities_primary, membrane_xs_primary, "Jet", None)
            heatmap.write_html(os.path.join(result_dir_primary, "heatmap.html"))
            heatmap.write_image(os.path.join(result_dir_primary, "heatmap.svg"))
            # save into temp directory
            save_output(output_primary, TMP_PRIMARY_DIR)

            # repeat for secondary video if exists
            if secondary_frames is not None and uploaded_file_secondary is not None:
                secondary_video_name, _ = os.path.splitext(uploaded_file_secondary.name)
                result_dir_secondary = os.path.join("data", directory, f"{time}_{secondary_video_name}")
                makedirs(result_dir_secondary)
                save_output(output_secondary, result_dir_secondary)
                write_config(config, os.path.join(result_dir_secondary, "config.toml"))
                heatmap = plot_membrane_heatmap(data_secondary, membrane_intensities_secondary, membrane_xs_secondary, "Jet", None)
                heatmap.write_html(os.path.join(result_dir_secondary, "heatmap.html"))
                heatmap.write_image(os.path.join(result_dir_secondary, "heatmap.svg"))
                save_output(output_secondary, TMP_SECONDARY_DIR)

        progress_bar.success("Measures are completed")

    primary_filelist = [f for f in os.listdir(TMP_PRIMARY_DIR)]
    secondary_filelist = [f for f in os.listdir(TMP_SECONDARY_DIR)]

    if len(primary_filelist) > 2:
        # show video
        video_slot.video(TMP_PRIMARY_OUTPUT_PATH)
        data_primary, membrane_intensities_primary, membrane_xs_primary, _ = read_output(TMP_PRIMARY_DIR)
        data_secondary, membrane_intensities_secondary, membrane_xs_secondary = None, None, None
        if uploaded_file_secondary and len(secondary_filelist) > 2:
            data_secondary, membrane_intensities_secondary, membrane_xs_secondary, _ = read_output(TMP_SECONDARY_DIR)

        # make plots
        st.header("Measures")

        window_size = st.slider("Bandwidth for smoothing curves with a moving average",
                                min_value=1,
                                max_value=10,
                                value=1,
                                step=1)

        aggregation_frames = st.slider("Number of frames to aggregate area growth with",
                                min_value=1,
                                max_value=15,
                                value=1,
                                step=1)

        growth_area_primary = plot_growth_area(data_primary, window_size, False, aggregation_frames)
        st.plotly_chart(
            plot_growth_area(data_primary, window_size, True, aggregation_frames),
            width="stretch"
        )

        colorscale = st.selectbox("Color",
                                  ("Viridis", "Inferno", "Plasma", "Jet",
                                   "Oranges", "solar", "deep", "Purp"),
                                  index=3)
        smooth = st.selectbox("Smoothing", ("None", "Fast", "Best"), index=0)
        other_figs_label = st.selectbox("Select graph to overlay",
                                        options=("None",
                                                 "Membrane mean intensity",
                                                 "Cytoplasm mean intensity",
                                                 "Growth area"),
                                        )

        membrane_intensity_primary = plot_membrane_intensity(
            data_primary, window_size, False)
        cytoplasm_intensity_primary = plot_cytoplasm_intensity(
            data_primary, window_size, False)
        labels_to_plot = {
            "Membrane mean intensity": membrane_intensity_primary,
            "Cytoplasm mean intensity": cytoplasm_intensity_primary,
            "Growth area": growth_area_primary
        }
        other_fig = labels_to_plot[other_figs_label] if other_figs_label != "None" else None
        heatmap_primary = plot_membrane_heatmap(
            data_primary, membrane_intensities_primary, membrane_xs_primary, colorscale, other_fig, smooth.lower())

        if uploaded_file_secondary and len(secondary_filelist) > 2:
            membrane_intensity_secondary = plot_membrane_intensity(
                data_secondary, window_size, False)
            cytoplasm_intensity_secondary = plot_cytoplasm_intensity(
                data_secondary, window_size, False)
            labels_to_plot = {
                "Membrane mean intensity": membrane_intensity_secondary,
                "Cytoplasm mean intensity": cytoplasm_intensity_secondary,
                "Growth area": growth_area_primary
            }
            other_fig = labels_to_plot[other_figs_label] if other_figs_label != "None" else None
            heatmap_secondary = plot_membrane_heatmap(
                data_secondary, membrane_intensities_secondary, membrane_xs_secondary, colorscale, other_fig, smooth.lower())

            primary_data_column, secondary_data_column = st.columns(2)
            with primary_data_column:
                st.subheader("Primary video measures")
                st.plotly_chart(
                    plot_membrane_intensity(data_primary, window_size, True),
                    width="stretch",
                )
                st.plotly_chart(
                    plot_cytoplasm_intensity(data_primary, window_size, True),
                    width="stretch"
                )
                st.plotly_chart(
                    heatmap_primary,
                    width="stretch"
                )
            with secondary_data_column:
                st.subheader("Secondary video measures")
                st.plotly_chart(
                    plot_membrane_intensity(data_secondary, window_size, True),
                    width="stretch",
                )
                st.plotly_chart(
                    plot_cytoplasm_intensity(data_secondary, window_size, True),
                    width="stretch"
                )
                st.plotly_chart(
                    heatmap_secondary,
                    width="stretch"
                )
        else:
            st.subheader("Primary video measures")
            st.plotly_chart(
                plot_membrane_intensity(data_primary, window_size, True),
                width="stretch",
            )
            st.plotly_chart(
                plot_cytoplasm_intensity(data_primary, window_size, True),
                width="stretch"
            )
            st.plotly_chart(
                heatmap_primary,
                width="stretch"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(e)
