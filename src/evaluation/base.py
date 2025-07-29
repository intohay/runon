from abc import ABC, abstractmethod

class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def prepare(self, reference_data):
        pass

    @abstractmethod
    def compute(self, hypothesis, reference_data=None):
        pass

    @abstractmethod
    def aggregate(self, scores):
        pass