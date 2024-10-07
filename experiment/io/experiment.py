import os

from ._filesystem import FileSystem, LocalFileSystem


class ExperimentData:
    def __init__(
            self,
            filesystem: FileSystem = LocalFileSystem(),
            data_path: str = "experiments"
    ):
        self.filesystem = filesystem
        self.root_path = data_path
        self.experiment_path = None
        self.results_dir = "results"
        self.conf_dir = "conf"

    def set_experiment_path(self, experiment_path):
        self.experiment_path = os.path.join(self.root_path, experiment_path)
