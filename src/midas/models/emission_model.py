from abc import ABC, abstractmethod

class EmissionModel(ABC):
    """TODO..."""
    @abstractmethod
    def get_spectra(self):
        raise NotImplementedError("Method get_spectra not implemented on child class")
