
from abc import ABC, abstractmethod


class RAG(ABC):
    
    @abstractmethod
    def retrieve_context(self, query, ): str
        pass
    
    
    