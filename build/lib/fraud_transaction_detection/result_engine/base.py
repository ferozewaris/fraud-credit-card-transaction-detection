from abc import abstractmethod


class ResultManager:

    @abstractmethod
    def _process_result(self, *args):
        pass

    def process_result(self, *args):
        return self._process_result(*args)