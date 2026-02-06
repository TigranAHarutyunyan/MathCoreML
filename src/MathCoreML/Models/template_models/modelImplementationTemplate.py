from abc import ABC, abstractmethod

class Models(ABC):
    @abstractmethod
    def preaper_data(self,x_column):
        pass
    @abstractmethod
    def train_model(self):
        pass
    @abstractmethod
    def predict_single(self,index:int):
        pass
    