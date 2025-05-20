from typing import List, Union, Optional, Literal
import dataclasses
import sys
sys.path.append('/home/zhouxueyang/Safety_code/')
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
# import openai

from openai import OpenAI

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


client = OpenAI(
    api_key="Your API key.",
    base_url="Base URL.",
)

import anthropic

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


# @retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    num_tokens=False,
    stop=None,
    seed: int = 1234,
) -> Union[List[str], str]:
    if stop:
        response = client.chat.completions.create(
            model=model,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            n=num_comps,
            seed=seed,
        )
    else:
        response = client.chat.completion.create(
            model=model,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            seed=seed,
        )

    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[dataclasses.asdict(message) for message in messages],
    #     max_tokens=max_tokens,
    #     temperature=temperature,
    #     top_p=1,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     n=num_comps,
    # )

    if num_comps == 1:
        if num_tokens:
            return [response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens]
        else:
            return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 8192, temperature: float = 0.2, num_comps: int = 1, num_tokens: bool = False, stop: list=None, seed: int = 1234) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1, num_tokens: bool=False, stop: list=None, seed: int=1234) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps, num_tokens, stop, seed=seed)


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")

class GPT4o(GPTChat):
    def __init__(self):
        super().__init__("gpt-4o")

class GPT4omini(GPTChat):
    def __init__(self):
        super().__init__("gpt-4o-mini")

class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo-16k")

class GPT432(GPTChat):
    def __init__(self):
        super().__init__("gpt-4-32k")

class Claude(GPTChat):
    def __init__(self):
        super().__init__("claude-3-5-sonnet-20240620")

class Qwen(GPTChat):
    def __init__(self, model_name):
        super().__init__(model_name)
        

class Gemini(GPTChat):
    def __init__(self, model_name):
        super().__init__(model_name)
    

class HFModelChat():
    def __init__(self, model_path, eos_token_id=None):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.is_chat = True


    def get_messages(self, messages):
        llama_messages = [
            dict(role="system",
                 content=messages[0].content
                 ),
            dict(role="user",
                 content=messages[1].content
                 ),
            ]

        return llama_messages

    def generate_chat(self, messages, max_tokens: int = 1024, temperature: float = 0.2, top_p=0.9):
        if temperature < 0.001:
            temperature = 0.001

        messages = self.get_messages(messages)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        out_ids = outputs[0][input_ids.shape[-1]:]
        outs = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        return outs

from typing import Dict, Union, Any
import tiktoken
OpenAISystemMessage = Dict[str, str]
OpenAIAssistantMessage = Dict[str, str]
OpenAIUserMessage = Dict[str, str]
OpenAIChatMessage = Union[OpenAIUserMessage, OpenAIAssistantMessage]
OpenAIMessage = Union[OpenAISystemMessage, OpenAIChatMessage]

def count_tokens_openai_chat_models(
        messages: List[OpenAIMessage],
        encoding: Any,
) -> int:
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += 1
    num_tokens += 2
    return num_tokens

def num_tokens_from_messages(
        messages: List[OpenAIMessage],
        model: str,
) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model("cl100k_base")

    return count_tokens_openai_chat_models(messages, encoding)

def num_tokens_from_prompt(
        prompt: str,
        model: str,
) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model("cl100k_base")

    num_tokens = len(encoding.encode(prompt))

    return num_tokens

def prompt_cost(model: str, num_prompt_tokens: float, num_completion_tokens: float):
    input_cost_map = {
        "gpt-3.5-turbo": 0.0015,
        "gpt-3.5-turbo-16k": 0.003,
        "gpt-4": 0.03,
        "gpt-4-32k": 0.06,
        "gpt-4-turbo": 0.01,
    }

    output_cost_map = {
        "gpt-3.5-turbo": 0.002,
        "gpt-3.5-turbo-16k": 0.004,
        "gpt-4": 0.06,
        "gpt-4-32k": 0.12,
        "gpt-4-turbo": 0.03,
    }

    return num_prompt_tokens * input_cost_map[model] / 1000.0 + num_completion_tokens * output_cost_map[model] / 1000.0