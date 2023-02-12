import dearpygui.dearpygui as dpg
import numpy as np
import math


def setup_scene():
    window_size = [1280, 720
    ]

    scene = generate_scene()

    # build the dpg scene
    with dpg.window(label="Scene", width=window_size[0], height=window_size[1], no_move=True, no_title_bar=True, no_collapse=True, no_close=True, no_background=True, no_focus_on_appearing=True, no_bring_to_front_on_focus=True, no_scrollbar=True, no_saved_settings=True):
        build_dpg_scene(scene, width=window_size[0], height=window_size[1])

    # setup the scene transforms
    dpg.apply_transform("scene-root", dpg.create_translation_matrix([250, 250]))
    apply_scene_transforms(scene)

    # show the window
    dpg.show_viewport()
    # dpg.set_primary_window("Scene", True)

    return scene


def generate_scene():
    base_planet = { "distance": 0, "angle": 0.0, "moons": []}
    default_config = { "distance": (0, 200), "angle": (0.0, 360.0), "moons": []}

    random_planet = lambda : { "distance": np.random.randint(*default_config["distance"]), 
                                  "angle": np.random.randint(*default_config["angle"]), 
                                  "moons":  [ [np.random.randint(5, 45), np.random.randint(0, 360)] for _ in range(np.random.randint(1, 5)) ] }

    scene = { f"planet{i}": random_planet() for i in range(10) }

    return scene


def build_dpg_scene(scene, width=500, height=500):
    """ Build the scene inside a dpg.window context. """
    with dpg.drawlist(width=width, height=height):
        with dpg.draw_node(tag="scene-root"):
            dpg.draw_circle([0, 0], 15, color=[255, 255, 0], fill=[255, 255, 0]) # sun

            # planets
            for pname, planet in scene.items():
                dpg.draw_circle([0, 0], planet["distance"], color=[0, 255, 0]) # orbit

                with dpg.draw_node(tag=pname):
                    dpg.draw_circle([0, 0], 10, color=[0, 255, 0], fill=[0, 255, 0]) # planet

                    for i, moon in enumerate(planet["moons"]):
                        dpg.draw_circle([0, 0], moon[0], color=[255, 0, 255]) # moon orbit

                        with dpg.draw_node(tag=f"{pname}-moon{i}"):
                            dpg.draw_circle([0, 0], 5, color=[255, 0, 255], fill=[255, 0, 255]) # moon


def apply_scene_transforms(scene):
    for name, planet in scene.items():
        dpg.apply_transform(name, dpg.create_rotation_matrix(math.pi*planet["angle"]/180.0 , [0, 0, -1])*dpg.create_translation_matrix([planet["distance"], 0]))

        for i, moon in enumerate(planet["moons"]):
            dpg.apply_transform(f"{name}-moon{i}", dpg.create_rotation_matrix(math.pi*moon[1]/180.0 , [0, 0, -1])*dpg.create_translation_matrix([moon[0], 0]))


def update_scene_transforms(scene, updated):
    for tag in updated:
        # decode tag into scene object
        if "-" in tag:
            a, b = tag.split("-moon")

            # apply parent transform
            dpg.apply_transform(a, dpg.create_rotation_matrix(scene[a]["angle"]/180.0 , [0, 0, -1])*dpg.create_translation_matrix([scene[a]["distance"], 0]))

            scene_obj = scene[a]["moons"][int(b)]
        else:
            scene_obj = scene[tag]["distance"], scene[tag]["angle"]

        # apply main transform
        dpg.apply_transform(tag, dpg.create_rotation_matrix(math.pi*scene_obj[1]/180.0 , [0, 0, -1])*dpg.create_translation_matrix([scene_obj[0], 0]))


def render_scene_loop(scene):
    while dpg.is_dearpygui_running():
        changed = []

        # update scene data
        scene["planet0"]["moons"][0][0] += 0.1
        # Todo: update on scene access
        changed.append("planet0-moon0")

        # TODO: tag updated values instead of full loop
        update_scene_transforms(scene, changed)

        dpg.render_dearpygui_frame()