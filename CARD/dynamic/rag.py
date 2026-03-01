from CARD.dynamic.dytool import Dytool
from CARD.dynamic.dytool_registry import ToolRegistry


@ToolRegistry.register("RAG")
class Dyrag(Dytool):
    def __init__(self):
        # Mode description mapping
        self.mode_info_dict = {
            "dense": "Dense RAG uses neural vector retrieval, suitable for semantic matching tasks, but with higher computational resource consumption.",
            "sparse": "Sparse RAG is based on keyword matching, efficient but may miss semantically relevant documents.",
            "hybrid": "Hybrid RAG combines dense and sparse methods, balancing recall and precision, suitable for comprehensive tasks.",
            "retriever_reranker": "Retriever + Reranker is a two-stage retrieval scheme that first retrieves then ranks, suitable for tasks requiring high precision.",
        }

        # Knowledge source description mapping
        self.source_info_dict = {
            "PDF": "PDF documents are typically structured or semi-structured files, suitable for handling formal documents, research reports, etc.",
            "Web": "Web pages have rich information but inconsistent structure, suitable for open-domain question answering and multi-source fusion tasks.",
            "Database": "Data in structured databases has high accuracy, suitable for numerical analysis, fact verification, etc.",
            "Wikipedia": "Wikipedia has broad coverage, suitable for encyclopedia-style and background knowledge supplementation tasks.",
            "SearchEngine": "Search engines serve as dynamic information sources, suitable for tasks requiring real-time and up-to-date information.",
        }

    def get_info_by_mode(self, mode: str) -> str:
        return self.mode_info_dict.get(
            mode, f"Unknown RAG mode: {mode}. Please check your input."
        )

    def get_info_by_source(self, source: str) -> str:
        return self.source_info_dict.get(
            source, f"Unknown knowledge source type: {source}. Please check your input."
        )

    def get_dynamic_info(self) -> str:
        return "Dyrag is a tool class for obtaining dynamic information related to RAG configuration. Use via `get_info_by_mode(mode)` and `get_info_by_source(source)` methods."
