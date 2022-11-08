from abc import abstractmethod


class PipelineDriver:

    @abstractmethod
    def run(self):
        pass

