import re
import sys
import argparse
# main.py 脚本路径
script_path = "main.py"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_model_name", type=str, default="gpt-4o")
    parser.add_argument("--agent_model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--adapter_model_path", type=str, default=None)
    parser.add_argument("--tool_path", type=str, default="./data/tools/all_toolkits.json")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_plan_step", type=int, default=15)
    parser.add_argument("--log_save_path", type=str, default="./log/")
    parser.add_argument("--run_pattern", type=str, default="test")
    parser.add_argument("--env_pattern", type=str, default="adv")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)

    args = parser.parse_args()
    return args

def args_to_list(args):
    arg_list = []
    for key, value in vars(args).items():
        if value is not None:
            arg_list.append(f"--{key}")
            arg_list.append(str(value))
    return arg_list


# 模拟执行脚本的逻辑
for s in range(1, 2):
    # 设置命令行参数
    # args = [
    #     "--env_model_name", "gpt-4o", "--agent_model_name", "gemini-1.5-pro", "--model_path", "/data/zxy/llama3.1_instruction",
    #     "--dataset_path", "/data/zxy/data/snapshot/ours/test_snapshots.json", "--tool_path",
    #     "./data/tools/all_toolkits.json", "--temperature", "0.8",
    #     "--max_plan_step", "15", "--log_save_path", "./log/", "--run_pattern", "sample", "--env_pattern", "adv",
    #     "--save_path", "/data/zxy/save/sample/ours/",
    #     "--sample_num", str(s)
    # ]

    # args = [
    #     "--env_model_name", "gpt-4o", "--agent_model_name", "qwen_100", "--model_path", "/data/zxy/glm_9b_chat", "--dataset_path", "/data/zxy/data/snapshot/public/test_snapshots.json",
    #     "--adapter_model_path", "/data/zxy/models/agent/glm_9b/100/", "--tool_path", "./data/tools/all_toolkits.json", "--temperature", "0.3",
    #     "--max_plan_step", "15", "--log_save_path", "./log/", "--run_pattern", "test", "--env_pattern", "adv",
    #     "--save_path", "/data/zxy/save/sample/glm/public/",
    #     "--sample_num", str(s)
    # ]

    """"--adapter_model_path", "/data/zxy/models/agent/llama_8b","""

    args = get_args()
    arg_list = args_to_list(args)

    arg_list.append("--sample")
    arg_list.append(str(s))

    # 模拟命令行参数 sys.argv
    sys.argv = [script_path] + arg_list

    # 执行 main.py 脚本并等待执行完成
    with open(script_path) as f:
        code = f.read()
        exec(code)

    # 输出结果
    print(f"Executed with sample_num={s}, waiting for the next round...")