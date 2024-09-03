#!/bin/bash

FLAGS="-pqh"

SCENE_DIR="scenes"
MEDIA_DIR="scenes/output"
mkdir -p $MEDIA_DIR

# for each scene in the scene dir, grab the class name if it matches `class <identifier>(Scene):`
for scene_py in $(ls $SCENE_DIR/*.py); do
    # find the classes in the scene file
    class_names=$(grep -oP 'class \K([a-zA-Z_][a-zA-Z0-9_]*)(?=\(Scene\):)' "$scene_py")

    # render each class
    for class_name in $class_names; do
        manim render $FLAGS $scene_py $class_name --media_dir $MEDIA_DIR --progress_bar display
    done
done
