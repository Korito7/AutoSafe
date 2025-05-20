import json
import os
import re
import os

def get_user_instruction_cost():
    data_path = "/data/zxy/save/snapshot/"
    paths = os.listdir(data_path)

    cases = []
    for path in paths:
        case_path = data_path + path
        case = json.load(open(case_path, encoding="utf-8"))
        cases.append(case)

    input_tokens = []
    output_tokens = []
    all_tokens = []
    all_costs = []
    for case in cases:
        input_tokens.append(case["cost_tokens"]["input_tokens"])
        output_tokens.append(case["cost_tokens"]["output_tokens"])
        all_tokens.append(case["cost_tokens"]["total_tokens"])

        cost = case["cost_tokens"]["input_tokens"] * 0.000003 + case["cost_tokens"]["output_tokens"] * 0.000015
        all_costs.append(cost)

    max_input_token = max(input_tokens)
    max_output_token = max(output_tokens)
    max_total_token = max(all_tokens)
    max_cost = max(all_costs)

    min_input_token = min(input_tokens)
    min_output_token = min(output_tokens)
    min_total_token = min(all_tokens)
    min_cost = min(all_costs)

    ave_input_token = sum(input_tokens) / len(input_tokens)
    ave_output_token = sum(output_tokens) / len(output_tokens)
    ave_total_token = sum(all_tokens) / len(all_tokens)
    ave_cost = sum(all_costs) / len(all_costs)

    print("Max input token: ", max_input_token)
    print("Max output token: ", max_output_token)
    print("Max total token: ", max_total_token)
    print("Max cost: ", max_cost)

    print("Min input token: ", min_input_token)
    print("Min output token: ", min_output_token)
    print("Min total token: ", min_total_token)
    print("Min cost: ", min_cost)

    print("Ave input token: ", ave_input_token)
    print("Ave output token: ", ave_output_token)
    print("Ave total token: ", ave_total_token)
    print("Ave cost: ", ave_cost)

    return


def get_trajectory_cost():
    data_path = "/data/zxy/save/snapshot/"
    paths = os.listdir(data_path)

    cases = []
    for path in paths:
        case_path = data_path + path
        case = json.load(open(case_path, encoding="utf-8"))
        cases.append(case)

    tokens = []
    costs = []
    for case in cases:
        snapshot_cost_tokens = case["snapshot_cost_tokens"]
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        for snapshot_cost_token in snapshot_cost_tokens:
            for s_c_t in snapshot_cost_token:
                if s_c_t:
                    total_tokens += s_c_t["total_tokens"]
                    input_tokens += s_c_t["input_tokens"]
                    output_tokens += s_c_t["output_tokens"]
                if s_c_t:
                    total_tokens += s_c_t["total_tokens"]
                    input_tokens += s_c_t["input_tokens"]
                    output_tokens += s_c_t["output_tokens"]

        tokens.append(total_tokens)
        cost = input_tokens * 0.000003 + output_tokens * 0.000015
        costs.append(cost)

    max_token = max(tokens)
    min_token = min(tokens)
    ave_token = sum(tokens) / len(tokens)


    max_cost = max(costs)
    min_cost = min(costs)
    ave_cost = sum(costs) / len(costs)

    print("Max token: ", max_token)
    print("Min token: ", min_token)
    print("Ave token: ", ave_token)

    print("Max cost: ", max_cost)
    print("Min cost: ", min_cost)
    print("Ave cost: ", ave_cost)

    return






def get_total_cost():
    data_path = "/data/zxy/save/snapshot/"
    paths = os.listdir(data_path)

    cases = []
    for path in paths:
        case_path = data_path + path
        case = json.load(open(case_path, encoding="utf-8"))
        cases.append(case)

    all_tokens = []
    all_costs = []
    for case in cases:
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0

        case_tokens = case["cost_tokens"]["total_tokens"]
        total_tokens += case_tokens
        input_tokens += case["cost_tokens"]["input_tokens"]
        output_tokens += case["cost_tokens"]["output_tokens"]

        snapshot_cost_tokens = case["snapshot_cost_tokens"]
        for snapshot_cost_token in snapshot_cost_tokens:
            for s_c_t in snapshot_cost_token:
                if s_c_t:
                    total_tokens += s_c_t["total_tokens"]
                    input_tokens += s_c_t["input_tokens"]
                    output_tokens += s_c_t["output_tokens"]
                if s_c_t:
                    total_tokens += s_c_t["total_tokens"]
                    input_tokens += s_c_t["input_tokens"]
                    output_tokens += s_c_t["output_tokens"]

        all_tokens.append(total_tokens)
        cost = input_tokens * 0.000003 + output_tokens * 0.000015
        all_costs.append(cost)

    max_token = max(all_tokens)
    min_token = min(all_tokens)
    ave_token = sum(all_tokens) / len(all_tokens)

    max_cost = max(all_costs)
    min_cost = min(all_costs)
    ave_cost = sum(all_costs) / len(all_costs)

    print("Max token: ", max_token)
    print("Min token: ", min_token)
    print("Ave token: ", ave_token)

    print("Max cost: ", max_cost)
    print("Min cost: ", min_cost)
    print("Ave cost: ", ave_cost)

    return


def main():
    get_user_instruction_cost()

    # get_trajectory_cost()

    get_total_cost()

    return


if __name__ == '__main__':
    main()