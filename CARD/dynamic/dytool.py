from typing import Dict, Any
from abc import ABC, abstractmethod

class Dytool(ABC):
    @abstractmethod
    def get_dynamic_info(self) -> str:
        pass
    
    @abstractmethod
    def get_info_by_mode(self, mode: str) -> str:
        pass

    @abstractmethod
    def get_info_by_source(self, source: str) -> str:
        pass