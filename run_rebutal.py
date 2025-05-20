import re
import sys

# main.py 脚本路径
script_path = "main.py"

# 模拟执行脚本的逻辑
for s in range(0, 1):
    # 设置命令行参数
    # args = [
    #     "--env_model_name", "gpt-4o", "--agent_model_name", "gemini-1.5-pro", "--model_path", "/data/zxy/llama3.1_instruction",
    #     "--dataset_path", "/data/zxy/data/snapshot/ours/test_snapshots.json", "--tool_path",
    #     "./data/tools/all_toolkits.json", "--temperature", "0.8",
    #     "--max_plan_step", "15", "--log_save_path", "./log/", "--run_pattern", "sample", "--env_pattern", "adv",
    #     "--save_path", "/data/zxy/save/sample/ours/",
    #     "--sample_num", str(s)
    # ]
    args = [
        "--env_model_name", "gpt-4o", "--agent_model_name", "qwen_BO&PD_1", "--model_path",
        "/data/zxy/qwen2.5-7b_instruction", "--dataset_path", "/data/zxy/data/rebutal/ours/test/datas_just_BO&PD.json",
        "--tool_path", "./data/tools/all_toolkits.json", "--temperature", "0.8",
        "--max_plan_step", "15", "--log_save_path", "./log/", "--run_pattern", "test", "--env_pattern", "adv",
        "--save_path", "/data/zxy/save/rebutal/sample/ours/",
        "--sample_num", str(s)
    ]
    """"--adapter_model_path", "/data/zxy/models/agent/llama_8b","""

    # 模拟命令行参数 sys.argv
    sys.argv = [script_path] + args

    # 执行 main.py 脚本并等待执行完成
    with open(script_path) as f:
        code = f.read()
        exec(code)

    # 输出结果
    print(f"Executed with sample_num={s}, waiting for the next round...")