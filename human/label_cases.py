import re
import json
import os
import sys
import sys,os
sys.path.append(os.getcwd())

import copy
import random
from Security_agent_ReAct_2 import untils, generation
from Security_agent_ReAct_2.prompts import agent, evaluator


# cases_path = "../data/cases/0-40cases.json"
cases_path = "../gen_cases/cases.json"
data = json.load(open(cases_path))

tools_path = "../data/tools/all_toolkits.json"
tools = json.load(open(tools_path))
tool_dict = untils.get_tool_dict(tools)

agentR = agent.AgentR()
agentR_model = untils.model_factory("qwen-turbo")

executor_agent = evaluator.Execution()
env_model = untils.model_factory("qwen-turbo")

max_plan_step = 15
temperature = 0.3
sample_num = 3

save_path = "../data/risky_cases/"

def get_tool_names(toolkits):
    tool_names = dict()
    tool_description = []
    id = 0
    for toolkit in toolkits:
        for tool in toolkit["tools"]:
            tool_description.append(str(id) + ". " + tool["name"] + ": " + tool["summary"])
            tool_names[id] = tool["name"]
            id += 1
    return "\n".join(tool_description), tool_names

def get_tool_name(item, tools, tool_dict, ):
    toolkit_names = item["Toolkits"]
    toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in toolkit_names]
    toolkits = [tools[toolkit_index] for toolkit_index in toolkit_indexs]

    tool_description, tool_names = get_tool_names(toolkits)

    return tool_description, tool_names


def print_colour(txt, color=None):
    if color == 'red':
        result = f"\033[31m{txt}\033[0m"
    elif color == 'green':
        result = f"\033[32m{txt}\033[0m"
    elif color == 'yellow':
        result = f"\033[33m{txt}\033[0m"
    elif color == 'blue':
        result = f"\033[34m{txt}\033[0m"
    elif color == 'violet':
        result = f"\033[35m{txt}\033[0m"
    elif color == 'cyan':
        result = f"\033[36m{txt}\033[0m"
    elif color == 'gray':
        result = f"\033[37m{txt}\033[0m"
    elif color == 'black':
        result = f"\033[30m{txt}\033[0m"
    else:
        result = txt
    return result


human_data = []
cases = os.listdir(save_path)
num = len(cases)
for case_id, item in enumerate(data[143 : ], 143):
    user_instruction = item["User Instruction"]
    potential_risky_outcomes = "- \n".join(item["Potential Risky Outcomes"])
    excepted_achievement = "- \n".join(item["Expected Achievements"])
    for _ in range(sample_num):
        print(print_colour(f"\n*** Case information {case_id} ***\n", color="red"))
        print("User Instruction: {user_instruction}".format_map(dict(user_instruction=user_instruction)))
        print("Potential Risky Outcomes: {potential_risky_outcomes}".format_map(dict(potential_risky_outcomes=potential_risky_outcomes)))
        print("Expected Achievements: {expected_achievement}".format_map(dict(expected_achievement=excepted_achievement)))

        h_d = copy.deepcopy(item)
        intermediate_steps = [item]

        print(print_colour("\n*** continue current execute yes or no? (key '0' to continue, key '1' to exit): ***", color="red"))
        flag = input("flag: ")
        if flag == '1':
            h_d["exps"] = intermediate_steps[1:]

            print("*** please input current potential risky outcome: ***")
            risky = input("risky: ")
            current_riskys = risky.split(";")
            h_d["Potential Risky Outcomes"] = current_riskys

            human_data.append(h_d)

            d_save_path = save_path + "case" + str(case_id) + "_" + str(num) + ".json"
            untils.write_data(h_d, d_save_path)
            num += 1

            break

        step = 0
        scratchpad = "Input: " + item["User Instruction"] + "\n"
        intermediate_step_str = ""
        seed = random.randint(1, 10000)
        while step < max_plan_step:
            intermediate_steps.append([None, None, None]) # action, answer, observation
            agentR_input_args = untils.get_agent_input_args(item, tools, tool_dict)
            prompts = agentR.get_prompt_sample(agentR_input_args)
            system_prompt = prompts[0]
            user_prompt = prompts[1] + "\n" + intermediate_step_str + f"Thought {step + 1}: "
            cur_prompts = (system_prompt, user_prompt)

            response = generation.get_next_action(model=agentR_model, prompts=cur_prompts, temperature=temperature,
                                                  stop=[f"\nObservation", "\nInput"], seed=seed)
            cur_action = untils.extract_thought_action_and_input(llm_output=response, step=step)
            print(print_colour(f"\n*** Current Action {step+1} ***\n", color="red"))
            # untils.print_(cur_action)
            print(response)

            tool_description, tool_names = get_tool_name(item, tools, tool_dict, )
            print(print_colour("\nAvailable tools:", color="red"))
            print("{tool_name}".format_map(dict(tool_name=tool_description)))

            print(print_colour("\n*** modify action yes or no? (key '0' to modify, key '1' to skip): ***", color="red"))
            flag = input("flag: ")
            if flag == '0':
                print("\nplease input thought: ")
                input_thought = input("thought: ")
                if input_thought != "":
                    cur_action["thought"] = input_thought

                print("\nplease select tool: ")
                tool_id = input("tool id: ")
                if tool_id != "":
                    cur_action["tool"] = tool_names[int(tool_id)]

                print("\nplease input action input: ")
                input_action_input = input("action input: ")
                if input_action_input != "":
                    cur_action["tool_input"] = input_action_input

            intermediate_steps[-1][0] = cur_action

            if untils.FINAL_ANSWER_ACTION in cur_action["tool"]:
                break

            execute_input_args = untils.get_execute_input_args(item, tools, tool_dict, cur_action, scratchpad, step + 1,
                                                               intermediate_steps)
            prompts = executor_agent.get_adv_prompt(execute_input_args)
            system_prompt = prompts[0]
            user_prompt = prompts[1] + f"\nObservation {step + 1}: "
            user_prompt += "\n\nPlease note that the simulation results are presented strictly as required and do not give any other explanation or explanation."
            cur_prompts = (system_prompt, user_prompt)
            observation = generation.get_next_observation(model=env_model, prompts=cur_prompts, temperature=temperature,
                                                          stop=["\nAction"], seed=seed).replace("\n", " ")
            print(print_colour(f"\n*** Current Observation {step+1} ***\n", color="red"))
            print(observation)

            print(print_colour("\n*** modify observation yes or no? (key '0' to modify, key '1' to skip): ***", color="red"))
            flag = input("flag: ")
            if flag == '0':
                print("\nplease input observation: ")
                input_observation = input("observation: ")
                if input_observation != "":
                    observation = input_observation

            intermediate_steps[-1][2] = observation.strip()

            print(print_colour("\n*** continue current execute yes or no? (key '0' to continue, key '1' to exit): ***", color="red"))
            flag = input("flag: ")
            if flag == '1':
                h_d["exps"] = intermediate_steps[1 : ]

                print("*** please input current potential risky outcome: ***")
                risky = input("risky: ")
                current_riskys = risky.split(";")
                h_d["Potential Risky Outcomes"] = current_riskys

                human_data.append(h_d)

                d_save_path = save_path + "case" + str(case_id) + "_" + str(num) + ".json"
                untils.write_data(h_d, d_save_path)
                num += 1

                break

            if intermediate_steps[-1][0] != None:
                intermediate_step_str += f"Thought {step + 1}: {intermediate_steps[-1][0]['thought']}\nAction {step + 1}: {intermediate_steps[-1][0]['tool']}\nAction Input {step + 1}: {intermediate_steps[-1][0]['tool_input']}\n"
                scratchpad += f"Action {step + 1}: {intermediate_steps[-1][0]['tool']}\nAction Input {step + 1}: {intermediate_steps[-1][0]['tool_input']}\n"
            if intermediate_steps[-1][1] != None:
                intermediate_step_str += f"Answer {step + 1}: {intermediate_steps[-1][1]}\n"
                scratchpad += f"Answer {step + 1}: {intermediate_steps[-1][1]}\n"
            if intermediate_steps[-1][2] != None:
                intermediate_step_str += f"Observation {step + 1}: {observation}\n"
                scratchpad += f"Observation {step + 1}: {observation}\n"

            step += 1