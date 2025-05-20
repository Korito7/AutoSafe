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



def run_sample_api(dataset: list[dict],
        tools: list[dict],
        tool_dict: dict,
        model_name: str,
        env_model_name: str,
        reflector_model_name: str = None,
        expert_model_name: str = None,
        evaluation_model_path: str = None,
        evaluation_adapter_model_path: str = None,
        temperature: float = 0.0,
        max_plan_step: int = 10,
        sample_num: int = 0,
        save_path: str = "./",
):
    assert model_name == expert_model_name

    agentR_model = untils.model_factory(model_name)
    agentR = agent.AgentR()

    evaluation = reward.SafetyModel(model_path=evaluation_model_path, adapter_path=evaluation_adapter_model_path)

    reflector = policy.Reflection()
    reflector_model = untils.model_factory(reflector_model_name)

    if expert_model_name:
        expert_model = untils.model_factory(expert_model_name)
    else:
        expert_model = agentR_model

    expert = agent.Expert()

    env_model = untils.model_factory(env_model_name)
    executor = evaluator.Execution()

    num = sample_num
    sample_save_path = save_path + "expert"
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    dataset = dataset
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

        expert_data = copy.deepcopy(item)
        intermediate_step_str = untils.get_exp_scratchpad(exps)
        intermediate_steps = [copy.deepcopy(item)]
        intermediate_steps += exps
        step = len(exps)
        success_flag = True
        expert_exps = []
        while step < max_plan_step:
            intermediate_steps.append([None, None, None])
            expert_exps.append([None, None, None])
            agentR_input_args = untils.get_agent_input_args(item, tools, tool_dict)
            agentR_input_args["scratchpad"] = intermediate_step_str

            prompts = agentR.get_prompt_sample(agentR_input_args)
            system_prompt = prompts[0]
            user_prompt = prompts[1] + "\n" + intermediate_step_str + f"Thought {step + 1}: "
            cur_prompts = (system_prompt, user_prompt)

            max_try_num = 3
            try_num = 0
            while True and (try_num < max_try_num):
                response = generation.get_next_action(model=agentR_model, prompts=cur_prompts, temperature=temperature, stop=[f"\nObservation", "\nInput"])
                print(response)

                cur_action = untils.extract_thought_action_and_input(llm_output=response.replace("cat.", "cat ."), step=step)
                if (cur_action["tool"] in tool_names and cur_action["tool"] != "") or cur_action["tool"] == untils.FINAL_ANSWER_ACTION:
                    break
                try_num += 1

            if try_num >= max_try_num:
                success_flag = False
                break

            safety_input_args = untils.get_safety_model_input_args(item, cur_action, intermediate_steps[:-1])
            safety_label = evaluation.safety_label(safety_input_args)

            if safety_label == 0:
                print("\nExpert execute action ...\n")
                safety_cur_action = run_expert(reflector_model, expert_model, expert_model_name, evaluation, reflector, expert, intermediate_steps, tools, tool_dict, tool_names, cur_action, step)
                print("\nFinsh\n")
                if safety_cur_action != None:
                    cur_action = safety_cur_action
                else:
                    success_flag = False

            if success_flag == False:
                break

            intermediate_steps[-1][0] = cur_action
            expert_exps[-1][0] = cur_action

            if untils.FINAL_ANSWER_ACTION in cur_action["tool"]:
                expert_data["expert_exps"] = expert_exps
                break

            execute_input_args = untils.get_execute_input_args(item, tools, tool_dict, cur_action,
                                                               "Input: " + instruction + "\n" + intermediate_step_str, step+1, [])
            execute_prompts = executor.get_prompt(execute_input_args)

            system_prompt = execute_prompts[0]
            cur_obs = f"Observation {step + 1}: "
            user_prompt = execute_prompts[1] + "\n" + cur_obs + "\n\nPlease note that the simulation observation output should be the return of the current tool execution and must be presented exactly as required without any other explanation or explanation."

            cur_prompts = (system_prompt, user_prompt)

            response = generation.get_next_observation(model=env_model, prompts=cur_prompts, temperature=temperature,
                                                       stop=["\nAction"])
            observation = response.replace(cur_obs, "").strip()

            intermediate_steps[-1][2] = observation
            expert_exps[-1][2] = observation


            if intermediate_steps[-1][0] != None:
                intermediate_step_str += f"Thought {step + 1}: {intermediate_steps[-1][0]['thought']}\nAction {step + 1}: {intermediate_steps[-1][0]['tool']}\nAction Input {step + 1}: {intermediate_steps[-1][0]['tool_input']}\n"
            if intermediate_steps[-1][1] != None:
                intermediate_step_str += f"Answer {step+1}: {intermediate_steps[-1][1]}\n"
            if intermediate_steps[-1][2] != None:
                intermediate_step_str += f"Observation {step + 1}: {observation}\n"

            step += 1

        if success_flag == False:
            continue

        cur_save_path = sample_save_path + "/" + "expert_" + str(i) + ".json"
        # untils.write_data(expert_data, cur_save_path)

    return

def run_sample_llama(dataset: list[dict],
        tools: list[dict],
        tool_dict: dict,
        model_path: str,
        env_model_name: str,
        reflector_model_name: str = None,
        evaluation_model_path: str = None,
        evaluation_adapter_model_path: str = None,
        temperature: float = 0.0,
        max_plan_step: int = 5,
        sample_num: int = 0,
        save_path: str = "./",
        agent_pattern: str = "safety",
):
    agentR_model = load_model.Chat_Model(model_path)
    agentR = agent.AgentR()

    evaluation = reward.SafetyModel(model_path=evaluation_model_path, adapter_path=evaluation_adapter_model_path)

    reflector = policy.Reflection()
    reflector_model = untils.model_factory(reflector_model_name)

    expert_model = agentR_model

    expert = agent.Expert()

    env_model = untils.model_factory(env_model_name)
    executor = evaluator.Execution()

    num = "1"
    sample_save_path = save_path + "llama" + "_8b_" + "expert_" + num
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    dataset = dataset
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

        expert_data = copy.deepcopy(item)
        intermediate_step_str = untils.get_exp_scratchpad(exps)
        intermediate_steps = [copy.deepcopy(item)]
        intermediate_steps += exps
        step = len(exps)
        success_flag = True
        expert_exps = []
        while step < max_plan_step:
            intermediate_steps.append([None, None, None])
            expert_exps.append([None, None, None])
            agentR_input_args = untils.get_agent_input_args(item, tools, tool_dict)
            agentR_input_args["scratchpad"] = intermediate_step_str
            if agent_pattern == "sample":
                prompt = agentR.get_llama_prompt(agentR_input_args)
            else:
                prompt = agentR.get_llama_safety_prompt(agentR_input_args)
            prompt += "\nThought {index}: \nAction {index}: \nAction Input {index}: ".format_map(dict(index=step+1))

            max_try_num = 5
            try_num = 0
            while True and (try_num < max_try_num):
                response = agentR_model.get_response(prompt, temperature=temperature)
                print("Current Response: \n", response)

                cur_action = untils.extract_thought_action_and_input(llm_output=response.replace("cat.", "cat ."), step=step)
                if (cur_action["tool"] in tool_names and cur_action["tool"] != "") or cur_action["tool"] == untils.FINAL_ANSWER_ACTION:
                    break
                try_num += 1

            if try_num >= max_try_num:
                success_flag = False
                break

            safety_input_args = untils.get_safety_model_input_args(item, cur_action, intermediate_steps[:-1])
            safety_label = evaluation.safety_label(safety_input_args)

            if safety_label == 0:
                print("\nExpert execute action ...\n")
                safety_cur_action = run_expert(reflector_model, expert_model, None, evaluation, reflector, expert, intermediate_steps, tools, tool_dict, tool_names, cur_action, step)
                print("\nFinsh\n")
                if safety_cur_action != None:
                    cur_action = safety_cur_action
                else:
                    success_flag = False

            if success_flag == False:
                break

            intermediate_steps[-1][0] = cur_action
            expert_exps[-1][0] = cur_action

            if untils.FINAL_ANSWER_ACTION in cur_action["tool"]:
                expert_data["expert_exps"] = expert_exps
                break

            break

            execute_input_args = untils.get_execute_input_args(item, tools, tool_dict, cur_action,
                                                               "Input: " + instruction + "\n" + intermediate_step_str, step+1, [])
            execute_prompts = executor.get_prompt(execute_input_args)

            system_prompt = execute_prompts[0]
            cur_obs = f"Observation {step + 1}: "
            user_prompt = execute_prompts[1] + "\n" + cur_obs + "\n\nPlease note that the simulation observation output should be the return of the current tool execution and must be presented exactly as required without any other explanation or explanation."

            cur_prompts = (system_prompt, user_prompt)

            response = generation.get_next_observation(model=env_model, prompts=cur_prompts, temperature=temperature,
                                                       stop=["\nAction"])
            observation = response.replace(cur_obs, "").strip()
            print("Observation Output: ", observation)

            intermediate_steps[-1][2] = observation
            expert_exps[-1][2] = observation


            if intermediate_steps[-1][0] != None:
                intermediate_step_str += f"Thought {step + 1}: {intermediate_steps[-1][0]['thought']}\nAction {step + 1}: {intermediate_steps[-1][0]['tool']}\nAction Input {step + 1}: {intermediate_steps[-1][0]['tool_input']}\n"
            if intermediate_steps[-1][1] != None:
                intermediate_step_str += f"Answer {step+1}: {intermediate_steps[-1][1]}\n"
            if intermediate_steps[-1][2] != None:
                intermediate_step_str += f"Observation {step + 1}: {observation}\n"

            step += 1

        if success_flag == False:
            continue

        if "expert_exps" not in expert_data.keys():
            expert_data["expert_exps"] = expert_exps

        cur_save_path = sample_save_path + "/" + "expert_" + str(i) + ".json"
        untils.write_data(expert_data, cur_save_path)

    return


def run_expert(
        reflector_model: ModelBase,
        expert_model,
        expert_model_name: str,
        safety_model,
        reflector,
        expert,
        intermediate_steps,
        tools,
        tool_dict,
        tool_names,
        cur_action,
        step: int,
        max_iter_num: int = 5,

):
    item = intermediate_steps[0]
    safety_label = 0
    iter_num = 0
    safety_cur_action = copy.deepcopy(cur_action)
    while (safety_label == 0) and (iter_num < max_iter_num):
        reflect_input_args = untils.get_reflect_input_args(item, safety_cur_action, intermediate_steps)
        prompts = reflector.get_prompt(reflect_input_args)
        reflection = generation.get_next_reflection(model=reflector_model, prompts=prompts, temperature=0.5)

        expert_input_args = untils.get_expert_input_args(item, tools, tool_dict, reflection, safety_cur_action, intermediate_steps)
        prompts = expert.get_prompt(expert_input_args)

        max_try_num = 5
        try_mum = 0
        while try_mum < max_try_num:
            if expert_model_name:
                response = generation.get_next_action(model=expert_model, prompts=prompts, temperature=0.5)
            else:
                prompt = prompts[0] + "\n" + prompts[1]
                response = expert_model.get_response(prompt, temperature=0.3)

            print("Expert action: ", response)

            safety_cur_action = untils.extract_thought_action_and_input(llm_output=response, step=step)
            if (safety_cur_action["tool"] in tool_names and safety_cur_action["tool"] != "") or safety_cur_action["tool"] == untils.FINAL_ANSWER_ACTION:
                break

            try_mum += 1

        if try_mum >= max_try_num:
            return None

        safety_input_args = untils.get_safety_model_input_args(item, safety_cur_action, intermediate_steps[:-1])
        safety_label = safety_model.safety_label(safety_input_args)

        iter_num += 1

    if safety_label == 1:
        return safety_cur_action

    return None