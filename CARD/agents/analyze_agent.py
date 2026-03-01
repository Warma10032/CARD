from typing import List, Any, Dict
import re

from CARD.graph.node import Node
from CARD.agents.agent_registry import AgentRegistry
from CARD.llm.llm_registry import LLMRegistry
from CARD.prompt.prompt_set_registry import PromptSetRegistry
from CARD.tools.search.search_registry import SearchRegistry


def find_strings_between_pluses(text):
    return re.findall(r"\@(.*?)\@", text)


@AgentRegistry.register("AnalyzeAgent")
class AnalyzeAgent(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_size: str = "",
        external_tool_type: str = "",
        external_tool: str = "",
        external_source: str = "",
    ):
        super().__init__(
            id,
            "AnalyzeAgent",
            domain,
            llm_name,
            llm_size,
            external_tool_type,
            external_tool,
            external_source,
        )
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.search_summary = ""

    async def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ) -> List[Any]:
        """To be overriden by the descendant class"""
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = f"{self.constraint}"
        user_prompt = (
            f"The task is: {raw_inputs['task']}\n"
            if self.role != "Fake"
            else self.prompt_set.get_adversarial_answer_prompt(raw_inputs["task"])
        )
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            #  before is knoledgable expert，self is Searcher
            if (
                self.role == "Searcher"
                and info["role"] == "Knowlegable Expert"
                and self.external_tool_type == "Search"
            ):
                queries = find_strings_between_pluses(info["output"])
                search_engine = SearchRegistry.get(self.external_tool)
                print(f"{self.external_tool} {self.external_source} {queries}")
                search = await search_engine.search_batch(
                    queries=queries, site=self.external_source
                )
                if len(search):
                    self.search_summary = ".\n".join(search)
                    user_prompt += f"The key entities of the problem are explained as follows:{self.search_summary}"
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += (
            f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n"
            if len(spatial_str)
            else ""
        )
        user_prompt += (
            f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}"
            if len(temporal_str)
            else ""
        )
        return system_prompt, user_prompt

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(
            input, spatial_info, temporal_info
        )
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        """To be overriden by the descendant class"""
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(
            input, spatial_info, temporal_info
        )
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await self.llm.agen(message)
        if self.search_summary != "":
            response += f"\n\n{self.search_summary}"
            self.search_summary = ""
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response
