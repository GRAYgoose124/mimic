import numpy as np
import logging
import dearpygui.dearpygui as dpg

from mimic.models.sequential import Sequential
from mimic.layers.dense import Dense

from mimic.utils.data import xor_set


def viewer(model):
    with dpg.window(label="Network Viewer"):
        with dpg.drawlist():
            pass

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = Sequential([Dense(2),
                        Dense(4),
                        Dense(4),
                        Dense(1)])

    model.train(xor_set, epochs=10, learning_rate=0.01, momentum=0.01)

    viewer(model)


if __name__ == '__main__':
    main()

