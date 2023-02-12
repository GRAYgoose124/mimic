import dearpygui.dearpygui as dpg
from mimic.app.scene import setup_scene, render_scene_loop

def main():
    app()


def app():
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    scene = setup_scene()

    render_scene_loop(scene)
