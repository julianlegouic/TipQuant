import datetime
import os
import streamlit as st
import toml

from src.core.run import get_tubes, get_data
from src.ui.plot import (plot_cytoplasm_intensity, plot_growth_area, plot_membrane_heatmap,
                             plot_membrane_intensity, plot_direction_angle)
from src.ui.progressionbar import ProgressionBar
from src.ui.sidebar import (video_params_sidebar, model_sidebar, membrane_sidebar,
                                region_sidebar, advanced_sidebar)
from src.ui.utils import write_config, local_css
from src.utils import clear_directory, makedirs, save_frames, get_frames, save_output, read_video, read_output

st.set_option("deprecation.showfileUploaderEncoding", False)

MAIN_DIRECTORY = '/'.join(os.path.realpath(__file__).split('/')[:-3])  # tipQuant directory
CONFIG_PATH = os.path.join(MAIN_DIRECTORY, "src/config.toml")
DEFAULT_CONFIG = toml.load(CONFIG_PATH)
IMG_PATH = os.path.join(MAIN_DIRECTORY, "data/assets")
TEMP_DIR = os.path.join(MAIN_DIRECTORY, "temp")
TEMP_VIDEO_PATH = os.path.join(MAIN_DIRECTORY, "temp/temp_video.mp4")
TEMP_OUTPUT_PATH = os.path.join(MAIN_DIRECTORY, "temp/temp_video_output.mp4")
config = DEFAULT_CONFIG.copy()


def main():
    st.title("TipQUANT")
    local_css("./src/css/style.css")

    # Video
    uploaded_file = st.file_uploader("Select file", type=["mp4", "avi", "tif", "m4v"], key="fu")
    load_video = st.button("Load video")
    video_slot = st.empty()
    run = st.button("Run")

    # Sidebar
    video_params_sidebar(config, DEFAULT_CONFIG)
    st.sidebar.markdown("# Output")
    directory = st.sidebar.text_input("Output directory", "output")
    model_sidebar(config, DEFAULT_CONFIG, TEMP_VIDEO_PATH)
    membrane_sidebar(config, DEFAULT_CONFIG)
    region_name = region_sidebar(config, DEFAULT_CONFIG, IMG_PATH)
    advanced_sidebar(config, DEFAULT_CONFIG, IMG_PATH)

    # Action
    if uploaded_file and load_video:
        clear_directory(TEMP_DIR)
        with st.spinner("Loading video ..."):
            # Currently we don't have access to the uploaded file path so we have to save the video
            # in a temporary file which we know the localisation in order to load it properly
            # https://github.com/streamlit/streamlit/issues/896.
            # Moreover we don't know the extension of the file so a "tif" button option is provided
            # in case one would want to upload sequences of images.
            frames = get_frames(uploaded_file, TEMP_VIDEO_PATH)
            # The temporary file should be in codec H264 in order to be visualized in streamlit.
            # We use skvideo which make better use of ffmpeg than opencv for encoding.
            # https://discuss.streamlit.io/t/problem-with-displaying-a-recorded-video/3458/10
            save_frames(frames, TEMP_VIDEO_PATH)

    if os.path.exists(TEMP_VIDEO_PATH):
        video_slot.video(TEMP_VIDEO_PATH)

    if run:
        with st.spinner("Running ..."):
            progress_bar = ProgressionBar()
            frames = read_video(TEMP_VIDEO_PATH)
            assert len(frames) > 0

            # compute
            tubes = get_tubes(frames, config, progress_bar)
            output = get_data(frames, tubes, config, region_name, progress_bar)
            output_frames, measure_data = output
            data, membrane_intensities, membrane_xs, _, _, _, _ = measure_data
            save_frames(output_frames, TEMP_OUTPUT_PATH)

        with st.spinner("Saving ..."):
            # save into target directory
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_name, _ = os.path.splitext(uploaded_file.name)
            result_dir = os.path.join("data", directory, f"{time}_{video_name}")
            makedirs(result_dir)
            save_output(output, result_dir)
            write_config(config, os.path.join(result_dir, "config.toml"))
            heatmap = plot_membrane_heatmap(data, membrane_intensities, membrane_xs, "Jet", None)
            heatmap.write_html(os.path.join(result_dir, "heatmap.html"))
            heatmap.write_image(os.path.join(result_dir, "heatmap.svg"))

            # save into temp directory
            save_output(output, TEMP_DIR)

        progress_bar.success("Measures are completed")

    filelist = [f for f in os.listdir(TEMP_DIR)]

    if len(filelist) > 2:
        # show video
        video_slot.video(TEMP_OUTPUT_PATH)
        data, membrane_intensities, membrane_xs, _ = read_output(TEMP_DIR)

        # make plots
        st.header("Measures")

        window_size = st.slider("Bandwidth for smoothing curves with a moving average",
                                min_value=1,
                                max_value=10,
                                value=1,
                                step=1)
        membrane_intensity = plot_membrane_intensity(data, window_size, False)
        st.plotly_chart(
            plot_membrane_intensity(data, window_size, True),
            use_container_width=True,
        )
        cytoplasm_intensity = plot_cytoplasm_intensity(data, window_size, False)
        st.plotly_chart(
            plot_cytoplasm_intensity(data, window_size, True),
            use_container_width=True
        )
        growth_area = plot_growth_area(data, window_size, False)
        st.plotly_chart(
            plot_growth_area(data, window_size, True),
            use_container_width=True
        )
        direction_angle = plot_direction_angle(data, window_size, final=False)
        st.plotly_chart(
            plot_direction_angle(data, window_size, final=True),
            use_container_width=True
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
                                                 "Growth area",
                                                 "Direction angle"),
                                        )
        labels_to_plot = {
            "Membrane mean intensity": membrane_intensity,
            "Cytoplasm mean intensity": cytoplasm_intensity,
            "Growth area": growth_area,
            "Direction angle": direction_angle
        }
        other_fig = labels_to_plot[other_figs_label] if other_figs_label != "None" else None
        heatmap = plot_membrane_heatmap(data, membrane_intensities, membrane_xs, colorscale, other_fig, smooth.lower())
        st.plotly_chart(
            heatmap,
            use_container_width=True
        )


if __name__ == "__main__":
    main()
