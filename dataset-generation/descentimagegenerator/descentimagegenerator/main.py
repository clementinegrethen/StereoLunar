import numpy as np

from descentimagegenerator.config import DIGConfig
from argparse import ArgumentParser
from shutil import copy2
from descentimagegenerator.trajectoryrenderer import TrajectoryRenderer, ImageFormat
from descentimagegenerator.display import DisplayVideo


if __name__ == "__main__":
    argument_parser = ArgumentParser(
        prog="Absnav_simulator",
        description="Simulator for absnav3D sequences on celestial bodies",
    )
    argument_parser.add_argument("-c", "--config_file", required=True)
    #    argument_parser.add_argument("-m", "--make_map", action='store_true')
    argument_parser.add_argument("-d", "--display", action="store_true")

    args = argument_parser.parse_args()

    moon_absnav_config = DIGConfig.load_from_file(args.config_file)
    renderer = TrajectoryRenderer(moon_absnav_config)

    if args.display:
        float_display = DisplayVideo("float image")

    start_index = moon_absnav_config.output_config.image_range[0]
    end_index = moon_absnav_config.output_config.image_range[1]
    end_index = end_index if end_index > 0 else 2 ** 32 - 1

    assert renderer.get_trajectory_length() is not None
    if end_index > renderer.get_trajectory_length():
        end_index = renderer.get_trajectory_length()
        print(f"Rendering to end index {end_index}")

    if moon_absnav_config.output_config.copy_trajectory_file:
        copy2(
            moon_absnav_config.scene_config.trajectory_path,
            moon_absnav_config.output_config.output_directory,
        )

    frame_range = range(start_index, end_index, moon_absnav_config.output_config.step)
    try:
        from tqdm import tqdm

        frame_range = tqdm(frame_range)
    except ImportError:
        pass

    for i in frame_range:
        render_ret = renderer.render_frame(i)
        if not render_ret:
            print(f"Could not render frame {i}")
            break

        if args.display:
            float_image = renderer.get_visible_frame(ImageFormat.Gray32F)
            float_display(float_image, renderer.get_depth_map())

        img_name = f"image{i:07d}.png"
        if moon_absnav_config.output_config.output_directory:
            img_name = f"{moon_absnav_config.output_config.output_directory}/{img_name}"
        renderer.save_visible_frame(img_name)
