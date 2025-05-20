import re
import pickle
import json
import sys
sys.path.append('/home/zhouxueyang/Safety_code/')

from Security_agent_ReAct_2.chat_model import GPT4, GPT35, GPT4o, Claude, GPT4omini, Qwen, Gemini


FINAL_ANSWER_ACTION = "Final Answer"
RISKY_TYPES = ["Privacy Leakage", "Computer Security", "Financial Loss", "Property Damage", "Physical Health", "Data Loss", "Illegal Activities", "Ethics & Morality", "Bias & Offensiveness"]
RISKY_DESCRIPTION = {
    "Privacy Leakage": "Improper handling or exposure of sensitive user information, such as personal identity, passwords, and location data, leading to privacy violations.",
    "Computer Security": "Agents performing unverified operations, such as downloading malicious files or running dangerous code, can cause system vulnerabilities or be subjected to network attacks.",
    "Financial Loss": "The wrong operation of the Agent causes financial damage to the user, such as incorrect payment, loss of investment opportunities, or disclosure of financial information.",
    "Property Damage": "The Agent causes damage to hardware, devices, or other physical property, such as misoperating robot hardware or devices.",
    "Physical Health": "Agents perform tasks that directly or indirectly threaten the safety of the human body, such as poor decisions in the medical field or mistakes in autonomous driving.",
    "Data Loss": "The Agent mistakenly deletes or overwrites important files, data, or fails to back up user data properly.",
    "Illegal Activities": "Agents are exploited or misused to perform actions prohibited by law, such as generating inappropriate content, disseminating illegal information, or assisting in criminal activity.",
    "Ethics & Morality": "Agents engage in actions that violate social ethics or moral norms, such as suggesting unethical decisions or violating basic human rights.",
    "Bias & Offensiveness": "The Agent's behavior or output reflects discriminatory bias or generates offensive or inappropriate content, which may harm the user experience or cause controversy.",
}


# read .json data
def read_data(path: str) -> list[dict]:
    # 读取JSON文件
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def read_data_pkl(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    f.close()

    return data

def write_data(data, path: str):
    with open(path, "w", encoding="utf-8") as f_write:
        json.dump(data, f_write, indent=4)
    f_write.close()
    return


def write_data_pkl(data, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
    return

# get tool dict
def get_tool_dict(tools: list[dict]):
    tool_dict = dict()
    for i, tool in enumerate(tools):
        toolkit = tool["toolkit"]
        tool_dict[toolkit] = i

    return tool_dict



def model_factory(model_name: str):
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name == "gpt-4o":
        return GPT4o()
    elif model_name == "gpt-4o-mini":
        return GPT4omini()
    elif model_name == "claude":
        return Claude()
    elif "qwen" in model_name.lower():
        return Qwen(model_name=model_name)
    elif "gemini" in model_name.lower():
        return Gemini(model_name=model_name)
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        raise ValueError(f"not exist codellama model!")
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def insert_indent(s: str, indent: str = "\t", insert_first=True) -> str:
    prefix = indent if insert_first else ""
    return prefix + s.rstrip("\n ").replace("\n", "\n" + indent) + "\n"


def get_toolkit_summary(toolkit) -> str:
    return f"<{toolkit['toolkit']}>: {toolkit['description_for_model']}\n"


def create_str(
    data, include_desc: bool = True, indent: str = "\t", level="medium"
) -> str:
    name = data["name"]
    prefix = f"{indent}- " if include_desc else ""
    desc = f": {data['description']}" if include_desc else ""
    type = ""
    if "type" in data:
        type = data["type"]
        if not data.get("required", True):
            type += ", optional"
        type = f" ({type})"
    if level == "medium":
        return f"{prefix}{name}{type}"
    else:
        return f"{prefix}{name}{type}{desc}"


def create_description(tool, level="medium") -> str:
    if level not in ["none", "low", "medium", "high"]:
        raise ValueError(
            f"Level must be one of 'none', 'low', 'medium', 'high'"
        )

    include_args_returns = level in ["low", "medium", "high"]
    include_desc = level in ["medium", "high"]
    include_exceptions = level in ["high"]

    def get_combined_desc(name, items):
        prefix = f" {name}"
        item_strs = [create_str(item, include_desc=include_desc, level=level) for item in items]
        if include_desc:
            desc = prefix + "\n" + "\n".join(item_strs) + "\n"
        else:
            desc = prefix + " " + ", ".join(item_strs) + "\n"
        return desc

    desc = tool["summary"].rstrip("., ") + "\n"

    if include_args_returns:
        desc += get_combined_desc("Arguments", tool["parameters"])
        desc += get_combined_desc("Returns", tool["returns"])
        if include_exceptions and "exceptions" in tool.keys():
            desc += get_combined_desc("Exceptions", tool["exceptions"])

    return desc


def create_tool_description(toolkit: dict, level="high"):
    desc = get_toolkit_summary(toolkit)
    desc += "Tool APIs:\n"
    for tool in toolkit["tools"]:
        tool_name = tool["name"]
        tool_desc = create_description(tool, level)
        tool_desc = insert_indent(tool_desc, insert_first=False)

        desc += f"\t* {tool_name}: {tool_desc}"
    return desc


def get_tool_names(toolkits) -> str:
    tool_names = []
    for toolkit in toolkits:
        for tool in toolkit["tools"]:
            tool_names.append(tool["name"])
    return ", ".join(tool_names)


def get_agent_input_args(item: dict, tools: list[dict], tool_dict: dict):
    toolkit_names = item["Toolkits"]
    toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in toolkit_names]
    toolkits = [tools[toolkit_index] for toolkit_index in toolkit_indexs]

    toolkit_strings = "\n".join(
        [create_tool_description(toolkit) for toolkit in toolkits]
    )

    expected_achievements = "-" + "\n- ".join(item["Expected Achievements"])
    potential_risky_outcomes = "-" + "\n- ".join(item["Potential Risky Outcomes"])

    tool_names = get_tool_names(toolkits)

    input_args = dict(
        tool_description=toolkit_strings,
        tool_name=tool_names,
        input=item["User Instruction"],
        expected_achievements=expected_achievements,
        potential_risky_outcomes=potential_risky_outcomes,
    )

    return input_args


def get_execute_input_args(item: dict, tools: list[dict], tool_dict: dict, cur_action: dict, scratchpad: str, step: int, intermediate_steps: list):
    toolkit_names = item["Toolkits"]
    toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in toolkit_names]
    toolkits = [tools[toolkit_index] for toolkit_index in toolkit_indexs]

    toolkit_strings = "\n".join(
        [create_tool_description(toolkit) for toolkit in toolkits]
    )

    cur_tool_name = cur_action["tool"]
    cur_tool_input = cur_action["tool_input"]
    flag = False
    for toolkit in toolkits:
        for tool in toolkit["tools"]:
            if tool["name"] == cur_tool_name:
                cur_tool = tool
                flag = True
                break
        if flag == True:
            break

    if flag == True:
        cur_tool_description = create_description(cur_tool, level="high")
    else:
        raise Exception(f"Current use tool not found!")

    potential_risky_outcome = "-" + "\n- ".join(item["Potential Risky Outcomes"])

    input_args = dict(
        toolkit_descriptions=toolkit_strings,
        current_tool=cur_tool_name,
        current_tool_input=cur_tool_input,
        current_tool_description=cur_tool_name + "\n" + cur_tool_description,
        potential_risky_outcome=potential_risky_outcome,
        step=step,
        scratchpad=scratchpad,
        input=item["User Instruction"],
    )

    return input_args


def extract_thought_action_and_input(llm_output: str, step: int):
    thought_label = "Thought"
    action_label = "Action"
    action_input_label = "Action Input"

    if thought_label not in llm_output:
        llm_output = thought_label + ": " + llm_output

    thought_start = llm_output.find(thought_label)
    if thought_start != -1:
        thought_end = llm_output.find(action_label, thought_start)  # 查找下一个部分的起点
        thought = llm_output[thought_start:thought_end].split(":", 1)[1].strip() if thought_end != -1 else \
        llm_output[thought_start:].split(":", 1)[1].strip()
    else:
        thought = ""

    action_start = llm_output.find(action_label)
    if action_start != -1:
        action_end = llm_output.find(action_input_label, action_start)  # 查找下一个部分的起点
        try:
            action = llm_output[action_start:action_end].split(":", 1)[1].strip() if action_end != -1 else llm_output[action_start:].split(":", 1)[1].strip()
        except Exception:
            action = ""
    else:
        action = ""

    input_start = llm_output.find(action_input_label)
    if input_start != -1:
        action_input = llm_output[input_start:].split(":", 1)[1].strip()
    else:
        action_input = ""

    # try:
    #     thought = llm_output.split(f"Action {step + 1}:")[0].strip()
    #     if "Action:" in thought:
    #         thought = thought.split("Action:")[0].strip()
    # except:
    #     print("Thought Error!")
    #     print(llm_output)
    #     thought = ""
    # try:
    #     action = llm_output.split(f"Action {step + 1}:")[1].split(f"Action Input {step + 1}:")[0].strip()
    # except:
    #     try:
    #         action = llm_output.split("Action:")[1].split("Action Input:")[0].strip()
    #     except:
    #         if FINAL_ANSWER_ACTION in llm_output.strip():
    #             action = FINAL_ANSWER_ACTION
    #         else:
    #             print("Action Error!")
    #             print(llm_output)
    #             action = ""
    # try:
    #     action_input = llm_output.split(f"Action Input {step + 1}:")[1].strip()
    # except:
    #     try:
    #         action_input = llm_output.split("Action Input:")[1].strip()
    #     except:
    #         if action != FINAL_ANSWER_ACTION:
    #             print("Action Input Error!")
    #             print(llm_output)
    #         action_input = ""

    thought_flag = f"Thought {step + 1}: "
    action_flag = f"Action {step + 1}: "
    action_input_flag = f"Action Input {step + 1}: "
    cur_action = dict(thought=thought.replace(thought_flag, ""),
                      tool=action.replace(action_flag, "").replace("[", "").replace("]", ""),
                      tool_input=action_input.replace(action_input_flag, ""))
    return cur_action


def get_instruction_as_key(cases):
    instruction_to_key = dict()
    for case in cases:
        instruction = case["User Instruction"]
        if instruction not in instruction_to_key.keys():
            instruction_to_key[instruction] = case

    return instruction_to_key


def get_exp_by_iter_num(exps, iter_num):
    for exp in exps:
        if len(exp["exp"]) == iter_num:
            return exp
    return None


def get_exp_scratchpad(exp, thought=True):
    scratchpad = ""
    for index, e in enumerate(exp):
        observation = e[2].replace("\n", "")
        if thought == True:
            thought_flag = f"Thought {index + 1}: "
            scratchpad += f"Thought {index + 1}: {e[0]['thought'].replace(thought_flag, '').strip()}\nAction {index + 1}: {e[0]['tool']}\nAction Input {index + 1}: {e[0]['tool_input']}\nObservation {index + 1}: {observation}\n"
        else:
            scratchpad += f"Action {index + 1}: {e[0]['tool']}\nAction Input {index + 1}: {e[0]['tool_input']}\nObservation {index + 1}: {observation}\n"

    return scratchpad


def multiple_sub_strings(input_string, sub_string, num):
    count = input_string.count(sub_string)
    return count > num


def get_safety_model_input_args(item: dict, cur_action: dict, intermediate_steps: list):
    instruction = item["User Instruction"]
    potential_risky_outcomes = "\n- ".join(item["Potential Risky Outcomes"])

    intermediate_step_str = get_exp_scratchpad(exp=intermediate_steps[1 : ], thought=False)
    cur_action = f"Action: {cur_action['tool']}\nAction Input: {cur_action['tool_input']}"

    args = {
        "id": "unlabel",
        "User Instruction": instruction,
        "Potential Risky Outcomes": potential_risky_outcomes,
        "Background Action": intermediate_step_str,
        "Current Action": cur_action,
    }

    return args

def get_reflect_input_args(item: dict, cur_action: dict, intermediate_steps: list):
    instruction = item["User Instruction"]
    expected_achievements = "\n- ".join(item["Expected Achievements"])
    potential_risky_outcomes = "\n- ".join(item["Potential Risky Outcomes"])

    scratchpad = ""
    for index, intermediate_step in enumerate(intermediate_steps[1:]):
        if intermediate_step[0] != None:
            thought_flag = f"Thought {index + 1}: "
            scratchpad += f"Thought {index + 1}: {intermediate_step[0]['thought'].replace(thought_flag, '').strip()}\nAction {index + 1}: {intermediate_step[0]['tool']}\nAction Input {index + 1}: {intermediate_step[0]['tool_input']}\nObservation {index + 1}: {intermediate_step[2]}\n"

    thought_flag = f"Thought {len(intermediate_steps[1:])}: "
    cur_action = f"Thought {len(intermediate_steps[1:])}: {cur_action['thought'].replace(thought_flag, '').strip()}\nAction {len(intermediate_steps[1:])}: {cur_action['tool']}\nAction Input {len(intermediate_steps[1:])}: {cur_action['tool_input']}"

    args = {
        "id": "unlabel",
        "User Instruction": instruction,
        "Expected Achievements": expected_achievements,
        "Potential Risky Outcomes": potential_risky_outcomes,
        "Background Action": scratchpad,
        "Current Action": cur_action
    }

    return args


def get_expert_input_args(item: dict, tools: list[dict], tool_dict: dict, reflection: str, cur_action: dict, intermediate_steps: list):
    toolkit_names = item["Toolkits"]
    toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in toolkit_names]
    toolkits = [tools[toolkit_index] for toolkit_index in toolkit_indexs]

    toolkit_strings = "\n".join(
        [create_tool_description(toolkit) for toolkit in toolkits]
    )
    tool_names = get_tool_names(toolkits)
    instruction = item["User Instruction"]
    expected_achievements = "\n".join(item["Expected Achievements"])
    potential_risky_outcomes = "\n".join(item["Potential Risky Outcomes"])

    scratchpad = ""
    for index, intermediate_step in enumerate(intermediate_steps[1:]):
        if intermediate_step[0] != None:
            thought_flag = f"Thought {index + 1}: "
            scratchpad += f"Thought {index + 1}: {intermediate_step[0]['thought'].replace(thought_flag, '').strip()}\nAction {index + 1}: {intermediate_step[0]['tool']}\nAction Input {index + 1}: {intermediate_step[0]['tool_input']}\nObservation {index + 1}: {intermediate_step[2]}\n"

    thought_flag = f"Thought {len(intermediate_steps[1:])}: "
    cur_action = f"Thought {len(intermediate_steps[1:])}: {cur_action['thought'].replace(thought_flag, '').strip()}\nAction {len(intermediate_steps[1:])}: {cur_action['tool']}\nAction Input {len(intermediate_steps[1:])}: {cur_action['tool_input']}"

    args = dict(
        tool_description=toolkit_strings,
        tool_name=tool_names,
        user_instruction=instruction,
        expected_achievements=expected_achievements,
        potential_risky_outcomes=potential_risky_outcomes,
        scratchpad=scratchpad,
        current_action=cur_action,
        reflection=reflection,
        step=len(intermediate_steps[1:])
    )

    return args


def get_label_input_args(item: dict):
    user_instruction = item["User Instruction"]
    potential_risky_outcomes = "\n- ".join(item["Potential Risky Outcomes"])
    background = get_exp_scratchpad(exp=item["exps"], thought=False)

    current_action = item["current action"]
    current_action_str = f"Action: {current_action['tool']}\nAction Input {current_action['tool_input']}"

    args = {
        "User Instruction": user_instruction,
        "Potential Risky Outcomes": potential_risky_outcomes,
        "Background Action": background,
        "Current Action": current_action_str
    }

    return args

def extract_expert_action(llm_output: str, step: int):
    try:
        thought = llm_output.split("Action:")[0].strip()
    except:
        print("Thought Error!")
        print(llm_output)
        thought = ""

    try:
        action = llm_output.split("Action:")[1].split("Action Input:")[0].strip()
    except:
        if FINAL_ANSWER_ACTION in llm_output.strip():
            action = FINAL_ANSWER_ACTION
        else:
            print("Action Error!")
            print(llm_output)
            action = ""

    try:
        action_input = llm_output.split("Action Input:")[1].strip()
    except:
        if action != FINAL_ANSWER_ACTION:
            print("Action Input Error!")
            print(llm_output)
        action_input = ""

    thought_flag = f"Thought: "
    action_flag = f"Action: "
    action_input_flag = f"Action Input: "
    cur_action = dict(thought=thought.replace(thought_flag, ""),
                      tool=action.replace(action_flag, ""),
                      tool_input=action_input.replace(action_input_flag, ""))

    return cur_action


def extract_evaluation(llm_output: str):
    thought = llm_output.split("[Assessment Thought]:")[1].split("[Risk Type]:")[0]
    risky = llm_output.split("[Risk Type]:")[1].split("[Assessment Label]:")[0]
    label = llm_output.split("[Assessment Label]:")[1]

    evaluation_outputs = dict(
        thought=thought.strip(),
        risky=risky.strip(),
        label=label.strip()
    )

    return evaluation_outputs


def extract_cases(user_instruction, expected_achievements, potential_risky_outcomes):
    cases = {"User Instruction": user_instruction,
             "Expected Achievements": expected_achievements,
             "Potential Risky Outcomes": potential_risky_outcomes}

    return cases


def get_action_by_api(prompts, ):



    return


def get_action_by_model(prompt, temperature, step, tool_names, model, max_try_num):
    try_num = 0
    while try_num < max_try_num:
        response = model.get_response(prompt, temperature=temperature)
        if "Observation" in response:
            response = response.split("Observation")[0]
        # else:
        #     if "\nThought" in response:
        #         response = response.split("\nThought")[0].strip()

        try:
            cur_action = extract_thought_action_and_input(llm_output=response, step=step)
            if (cur_action["tool"] in tool_names and cur_action["tool"] != "") or cur_action["tool"] == FINAL_ANSWER_ACTION:
                if cur_action["tool"] != FINAL_ANSWER_ACTION and FINAL_ANSWER_ACTION in cur_action["tool_input"]:
                    cur_action["tool_input"] = cur_action["tool_input"].replace(FINAL_ANSWER_ACTION, "").strip()
                return cur_action
            else:
                try_num += 1
                continue
        except:
            try_num += 1
            continue

    return None

def get_action_by_api(model_name, messages, temperature, step, tool_names, agentR_model, max_try_num):
    try_num = 0
    while try_num < max_try_num:
        response, _ = agentR_model.get_llm_response(model_name=model_name, messages=messages, temperature=temperature)
        if "Observation" in response:
            response = response.split("Observation")[0]
        # else:
        #     if "\nThought" in response:
        #         response = response.split("\nThought")[0].strip()

        try:
            cur_action = extract_thought_action_and_input(llm_output=response, step=step)

            if (cur_action["tool"] in tool_names and cur_action["tool"] != "") or cur_action["tool"] == FINAL_ANSWER_ACTION:
                if cur_action["tool"] != FINAL_ANSWER_ACTION and FINAL_ANSWER_ACTION in cur_action["tool_input"]:
                    cur_action["tool_input"] = cur_action["tool_input"].replace(FINAL_ANSWER_ACTION, "").strip()
                return cur_action
            else:
                try_num += 1
                continue
        except:
            try_num += 1
            continue


def print_(data):
    for k in data.keys():
        print(f"{k}: {data[k]}")

    return