import os
import pickle   
from pathlib import Path


class FileMixin:
    def save_checkpoint(self, filename):
        file = Path(filename)
        if file.exists():
            checkpoint_number = int(file.stem.split("_cp_")[-1])
            checkpoint_number += 1
            file = file.with_name(
                file.stem.split("_cp_")[0] + f"_cp_{checkpoint_number}"
            )
            filename = str(file)
        else:
            file = file.with_name(file.stem + "_cp_0")
            filename = str(file)
        self.save(filename)

    def find_checkpoints(self, directory):
        checkpoints = []
        for file in os.listdir(directory):
            if "_cp_" in file:
                checkpoints.append(file)
        return checkpoints

    def get_latest_checkpoint(self, directory):
        checkpoints = self.find_checkpoints(directory)
        if not checkpoints:
            return None
        latest_checkpoint = max(
            checkpoints, key=lambda x: os.path.getctime(os.path.join(directory, x))
        )
        return latest_checkpoint

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.layers, f)

    @classmethod
    def load(cls, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as f:
            obj = cls()
            layers = pickle.load(f)
            return layers
