import re
from Security_agent_ReAct_2.chat_model import Message, ModelBase


def generation_func_impl(
      model: ModelBase,
      input_prompts: tuple,
      temperature: float = 0.0,
      stop: list = None,
      seed: int = 1234,
):
    messages = [
        Message(
            role="system",
            content=input_prompts[0],
        ),
        Message(
            role="user",
            content=input_prompts[1],
        )
    ]
    outputs = model.generate_chat(messages=messages, temperature=temperature, stop=stop, seed=seed)

    return outputs


def get_next_action(
        model: ModelBase,
        prompts: (str, str),
        temperature: float = 0.0,
        stop: list = None,
        seed: int = 1234,

):
    llm_output = generation_func_impl(model=model,
                         input_prompts=prompts,
                         temperature=temperature,
                         stop=stop,
                         seed=seed)

    # print(llm_output)

    return llm_output


def get_next_observation(
        model: ModelBase,
        prompts: (str, str),
        temperature: float = 0.0,
        stop: list = None,
        seed: int = 1234,
):
    llm_output = generation_func_impl(model=model,
                                      input_prompts=prompts,
                                      temperature=temperature,
                                      stop=stop,
                                      seed=seed)

    # print(llm_output)

    return llm_output

def get_next_reflection(
        model: ModelBase,
        prompts: (str, str),
        temperature: float = 0.0,
        stop: list = None,
):
    llm_output = generation_func_impl(model=model,
                                      input_prompts=prompts,
                                      temperature=temperature,
                                      stop=stop)

    print(llm_output)

    return llm_output


def get_next_evaluation(
        model: ModelBase,
        prompts: (str, str),
        temperature: float = 0.0,
        stop: list = None,
        seed: int = 1234,
):
    llm_output = generation_func_impl(model=model,
                                      input_prompts=prompts,
                                      temperature=temperature,
                                      stop=stop,
                                      seed=seed)
    # print(llm_output)

    return llm_output

def get_next_case(
        model: ModelBase,
        prompts: (str, str),
        temperature: float = 0.0,
        stop: list = None,
        seed: int = 1234,
):
    llm_output = generation_func_impl(model=model,
                                      input_prompts=prompts,
                                      temperature=temperature,
                                      stop=stop,
                                      seed=seed)

    print(llm_output)

    return llm_output