import argparse
import datetime
import json
import os

from src.core.run import get_tubes, get_data
from src.utils import save_output, makedirs, read_video
from src.ui.plot import plot_membrane_heatmap, plot_membrane_intensity


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c",
                        "--config",
                        type=str,
                        help="path to JSON config file, see config example in src/config.json",
                        default="src/config.json"
                        )
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as json_file:
        configs = json.load(json_file)

    for config in configs["CONFIGS"]:
        # compute
        print(f'Working on: {config["VIDEO_PATH"]}')
        region_name = config["REGION_NAME"]
        frames = read_video(config["VIDEO_PATH"])
        print("Computing tubes...")
        tubes = get_tubes(frames, config)
        print("Computing data...")
        output = get_data(frames, tubes, config, region_name)
        output_frames, measure_data = output
        data, membrane_intensities, membrane_xs, _, _, _, _ = measure_data

        # dump
        print("Saving...")
        video_name, _ = os.path.splitext(config["VIDEO_PATH"].split("/")[-1])
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result_dir = os.path.join(config["OUTPUT_DIR"], f"{timestamp}_{video_name}")
        makedirs(result_dir)
        with open(f'{result_dir}/config.json', 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file)
        save_output(output, result_dir)
        heatmap = plot_membrane_heatmap(data, membrane_intensities, membrane_xs, "Jet", None)
        heatmap.write_image(os.path.join(result_dir, "heatmap.svg"))
        membrane_intensity = plot_membrane_intensity(data, config["TIP"]["WINDOW_SIZE"], True)
        membrane_intensity.write_image(os.path.join(result_dir, "membrane_intensity.svg"))