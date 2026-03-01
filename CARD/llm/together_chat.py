import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from CARD.llm.format import Message
from CARD.llm.price import cost_count
from CARD.llm.llm import LLM
from CARD.llm.llm_registry import LLMRegistry
from together import Together


OPENAI_API_KEYS = [""]
BASE_URL = ""

load_dotenv()
MINE_BASE_URL = os.getenv("BASE_URL")
MINE_API_KEYS = os.getenv("API_KEY")



@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],
):
    client = Together(api_key=MINE_API_KEYS)
    response = client.chat.completions.create(
        model=model,
        messages=msg,
        frequency_penalty=0.1,
        max_tokens=1024,
        logit_bias=None,
        logprobs=None,
        n=1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        temperature=0.8,
        top_p=0.95,
    )

    choices_list = response.choices

    for c in choices_list:
        content = c.message.content
        prompt = "".join([item["content"] for item in msg])
        cost_count(prompt, content, "gpt-4")
        return content

    return "API access exception"


@LLMRegistry.register("TogetherChat")
class TogetherChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name, messages)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
