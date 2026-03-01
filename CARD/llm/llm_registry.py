from typing import Optional
from class_registry import ClassRegistry

from CARD.llm.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()
MY_SERVER = os.getenv("SERVER")


class LLMRegistry:
    registry = ClassRegistry()
    MODEL_NAME_MAP = {
        "siliconflow": {
            None: "Pro/deepseek-ai/DeepSeek-V3",
            "": "Pro/deepseek-ai/DeepSeek-V3",
            "deepseek-V3": "deepseek-ai/DeepSeek-V3",
            "deepseek-R1": "Pro/deepseek-ai/DeepSeek-R1",
            "QwQ-32B": "Qwen/QwQ-32B",
            "Qwen2.5-Coder-32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "qwen-7B": "Pro/Qwen/Qwen2.5-7B-Instruct",
            "qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
            "qwen-72B": "Qwen/Qwen2.5-72B-Instruct-128K",
            "qwen-7B-reasoner": "Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
        "together": {
            None: "deepseek-ai/DeepSeek-V3",
            "": "deepseek-ai/DeepSeek-V3",
            "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
            "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
            "llama-3-70B": "meta-llama/Llama-3-70b-chat-hf",
            "QwQ-32B-Preview": "Qwen/QwQ-32B-Preview",
            "Qwen2.5-Coder-32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "qwen-1.5B": "Qwen/Qwen2-1.5B",
            "qwen-7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "qwen-72B": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        },
        "api2d": {
            None: "gpt-4o-mini",
            "": "gpt-4o-mini",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4o": "gpt-4o",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4": "gpt-4-1106-preview",
            "Gemini-2.0-flash": "gemini-2.0-flash",
            "o1-mini": "o1-mini",
            "o3-mini": "o3-mini",
        },
        "deepseek": {
            "deepseek-V3": "deepseek-chat",
            "deepseek-R1": "deepseek-reasoner",
        },
        "qwen": {
            "qwen-14B": "qwen2.5-14b-instruct-1m",
            "qwen-7B": "qwen2.5-7b-instruct-1m",
        },
    }

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name == "llama-3-70B":
            model_name = cls.MODEL_NAME_MAP.get("together").get(model_name)
            model = cls.registry.get("TogetherChat", model_name)
            return model

        if model_name == "deepseek-V3" or model_name == "deepseek-R1":
            model_name = cls.MODEL_NAME_MAP.get("deepseek").get(model_name)
            model = cls.registry.get("GPTChat", model_name)
            return model

        if model_name == "qwen-72B":
            model_name = cls.MODEL_NAME_MAP.get("together").get(model_name)
            model = cls.registry.get("TogetherChat", model_name)
            return model

        model_name = cls.MODEL_NAME_MAP.get(MY_SERVER).get(model_name)

        if MY_SERVER == "together":
            model = cls.registry.get("TogetherChat", model_name)
        if MY_SERVER == "qwen":
            model = cls.registry.get("GPTChat", model_name)
        else:
            model = cls.registry.get("GPTChat", model_name)

        return model
