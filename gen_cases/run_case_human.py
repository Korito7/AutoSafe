import re
import os

import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_path)

from Security_agent_ReAct_2 import untils

model_cases_dir = "./cases/"
model_cases_paths = os.listdir(model_cases_dir)
save_path = "./human_cases/"
case_num = 83

for path in model_cases_paths[3 : ]:
    model_case_path = model_cases_dir + "/" + path
    model_case = untils.read_data(model_case_path)

    instruction = model_case["User Instruction"]
    expected_achievements = model_case["Expected Achievements"]
    potential_risky = model_case["Potential Risky Outcomes"]
    toolkits = model_case["Toolkits"]
    risky = model_case["Risky"]

    print("\nCurrent model case:\n")
    print("User Instruction: ", instruction)
    print()
    print("Toolkits: ", toolkits)
    print()
    print("Expected Achievements: ", expected_achievements)
    print()
    print("Potential Risky Outcomes: ", potential_risky)
    print()
    print("Risky: ", risky)
    print()

    flag = input("please select revise or skip? 1 is revise, 0 is skip, please input: ")
    if flag == "0":
        continue

    print("Now, starting revise case ...\n")
    print("User Instruction: ", instruction)

    flag = input("revise yes or no? 1 is yes, 0 is no, please input: ")
    if flag == "1":
        instruction = input("please input instruction: ")

    print("Toolkits: ", toolkits)

    flag = input("revise yes or no? 1 is yes, 0 is no, please input: ")
    if flag == "1":
        toolkits = input("please input tool: ")
        toolkits = toolkits.split(";")

    print("Expected Achievements: ", expected_achievements)
    flag = input("revise yes or no? 1 is yes, 0 is no, please input: ")
    if flag == "1":
        expected_achievements = input("please input expected achievements: ")
        expected_achievements = expected_achievements.split(";")
    else:
        expected_achievements = [expected_achievements]

    print("Potential Risky Outcomes: ", potential_risky)
    flag = input("revise yes or no? 1 is yes, 0 is no, please input: ")
    if flag == "1":
        potential_risky = input("please input potential risky: ")
        potential_risky = potential_risky.split(";")
    else:
        potential_risky = [potential_risky]

    print("Risky: ", risky)
    flag = input("revise yes or no? 1 is yes, 0 is no, please input: ")
    if flag == "1":
        risky = input("please input risky: ")
        risky = risky.split(";")
    else:
        risky = [risky]

    human_case = dict()

    human_case["User Instruction"] = instruction
    human_case["Toolkits"] = toolkits
    human_case["Expected Achievements"] = expected_achievements
    human_case["Potential Risky Outcomes"] = potential_risky
    human_case["Risky"] = risky

    save_case_path = save_path + "case_" + str(case_num) + ".json"
    untils.write_data(human_case, save_case_path)
    case_num += 1
