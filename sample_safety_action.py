import json
import re
import copy

import prompts.agent as agent
import prompts.evaluator as evaluator
import prompts.policy as policy
from chat_model import ModelBase

import untils
import generation
import reward

import sys
import os
sys.path.insert(0, '/data/zxy/LLaMA-Factory-main/src')
import load_model



def run_sample_api(
        dataset: list[dict],
        tools: list[dict],
        tool_dict: dict,
        agent_model_name: str,
        log_save_path: str = "./",
        temperature: float = 0.0,
        max_plan_step: int = 10,
        sample_num: str = "0",
        save_path: str = "./",
        agent_pattern: str = "safety",

):
    agentR_model = untils.model_factory(agent_model_name)
    agentR = agent.AgentR()

    evaluation = reward.SafetyModel(model_path="/data/zxy/llama3.1_instruction", adapter_path="/data/zxy/models/safety")

    num_success = 0
    item_num = len(dataset)
    if item_num > 50:
        dataset = dataset[: 50]

    num = sample_num
    sample_save_path = save_path + agent_model_name + "_" + num
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    sample_log_save_path = log_save_path + agent_model_name + "_" + num
    if not os.path.exists(sample_log_save_path):
        os.makedirs(sample_log_save_path)

    for i, item in enumerate(dataset):
        instruction = item["User Instruction"]
        print("User Instruction: ", instruction)
        exps = item["exps"]
        print("Execute: ")
        for exp in exps:
            untils.print_(exp[0])

        toolkit_names = item["Toolkits"]
        toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in toolkit_names]
        toolkits = [tools[toolkit_index] for toolkit_index in toolkit_indexs]

        tool_names = []
        for toolkit in toolkits:
            for tool in toolkit["tools"]:
                tool_names.append(tool["name"])

        action_data = copy.deepcopy(item)

        intermediate_step_str = untils.get_exp_scratchpad(exps)
        intermediate_steps = [copy.deepcopy(item)]
        intermediate_steps += exps
        step = len(exps)
        while step < max_plan_step:
            intermediate_steps.append([None, None, None])
            agentR_input_args = untils.get_agent_input_args(item, tools, tool_dict)
            prompts = agentR.get_prompt(agentR_input_args)
            system_prompt = prompts[0]
            # user_prompt = prompts[1] + "\n" + intermediate_step_str + f"Thought {step+1}: "

            # claude model
            user_prompt = (prompts[1] + "\n" + intermediate_step_str
                           + f"Thought {step+1}: " + "\n"
                           + f"Action {step+1}: " + "\n"
                           + f"Action Input {step+1}: " + "\n"
                           + "Note that you only need to give the output of the current round, which contains: a Thought, an Action and an Action Input, don't try to give all subsequent outputs, and don't include any explanations in your output.")

            cur_prompts = (system_prompt, user_prompt)

            max_try_num = 3
            try_num = 0
            while True and (try_num < max_try_num):
                response = generation.get_next_action(model=agentR_model, prompts=cur_prompts, temperature=temperature, stop=[f"\nObservation", "\nInput"])
                print(response)
                stop = "\nObservation"
                response = response.split(stop)[0]

                cur_action = untils.extract_thought_action_and_input(llm_output=response, step=step)

                safety_input_args = untils.get_safety_model_input_args(item, cur_action, intermediate_steps[:-1])
                safety_label = evaluation.safety_label(safety_input_args)

                if cur_action["tool"] in tool_names or cur_action["tool"] == untils.FINAL_ANSWER_ACTION:
                    break
                try_num += 1

            action_data["current_action"] = cur_action
            action_data["label"] = str(safety_label)

            step += 1
            break

        if try_num >= max_try_num:
            continue

        cur_save_path = sample_save_path + "/" + "action_" + str(i) + ".json"
        untils.write_data(action_data, cur_save_path)

    print("Case num: {item_num}".format_map(dict(item_num=item_num)))
    print("Success case num: {num_success}".format_map(dict(num_success=num_success)))

    return

def run_sample_llama(dataset: list[dict],
        tools: list[dict],
        tool_dict: dict,
        model_path: str,
        adapter_model_path: str = None,
        log_save_path: str = "./",
        temperature: float = 0.0,
        max_plan_step: int = 10,
        sample_num: int = 0,
        save_path: str = "./",
        agent_pattern: str = "safety",
):
    agentR_model = load_model.Chat_Model(model_path, adapter_model_path)
    agentR = agent.AgentR()

    num = sample_num
    sample_save_path = save_path + "llama_8b" + "_" + "naive_1_" + num
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    sample_log_save_path = log_save_path + "llama_8b" + "_" + "naive_1_" + num
    if not os.path.exists(sample_log_save_path):
        os.makedirs(sample_log_save_path)

    dataset = dataset[ : 50]
    for i, item in enumerate(dataset):
        instruction = item["User Instruction"]
        print("User Instruction: ", instruction)

        print("Execute: ")
        exps = item["exps"]
        for exp in exps:
            untils.print_(exp[0])

        toolkit_names = item["Toolkits"]
        toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in toolkit_names]
        toolkits = [tools[toolkit_index] for toolkit_index in toolkit_indexs]

        tool_names = []
        for toolkit in toolkits:
            for tool in toolkit["tools"]:
                tool_names.append(tool["name"])

        action_data = copy.deepcopy(item)
        intermediate_step_str = untils.get_exp_scratchpad(exps)
        step = len(exps)
        while step < max_plan_step:
            agentR_input_args = untils.get_agent_input_args(item, tools, tool_dict)
            agentR_input_args["scratchpad"] = intermediate_step_str

            prompt = agentR.get_llama_safety_prompt(agentR_input_args)
            prompt += "\nThought {index}: \nAction {index}: \nAction Input {index}: ".format_map(dict(index=step+1))

            max_try_num = 5
            try_num = 0
            while True and (try_num < max_try_num):
                response = agentR_model.get_response(prompt, temperature=temperature)
                print(response)

                cur_action = untils.extract_thought_action_and_input(llm_output=response.replace("cat.", "cat ."), step=step)
                if cur_action["tool"] in tool_names or cur_action["tool"] == untils.FINAL_ANSWER_ACTION:
                    break
                try_num += 1

            action_data["current_action"] = cur_action
            step += 1
            break

        if try_num >= max_try_num:
            continue

        cur_save_path = sample_save_path + "/" + "action_" + str(i) + ".json"
        untils.write_data(action_data, cur_save_path)

    return