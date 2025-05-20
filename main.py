import re
import json
import argparse

import os

import sys
sys.path.append('/home/zhouxueyang/Safety_code/')
import sample_risky_action as sample
import sample_risky_expert as expert
import sample_safety_action as safety_sample
import test
import untils

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_model_name", type=str, help="the env model name")
    parser.add_argument("--agent_model_name", type=str, help="the agent model name")
    parser.add_argument("--model_path", type=str, help="original model path if use local model")
    parser.add_argument("--adapter_model_path", type=str, help="adapter model path if use local model", default=None)
    parser.add_argument("--reflector_model_name", type=str, help="reflector model path if use local model", default=None)
    parser.add_argument("--expert_model_name", type=str, help="the expert model name", default=None)
    parser.add_argument("--dataset_path", type=str, help="The path of the dataset")
    parser.add_argument("--tool_path", type=str, help="The path of the tools.")
    parser.add_argument("--temperature", type=float, help="The temperature of model.")
    parser.add_argument("--max_plan_step", type=int, help="The max step of agent action in data.")
    parser.add_argument("--log_save_path", type=str, help="log save path.")
    parser.add_argument("--run_pattern", type=str, help="agent run pattern sample / expert / test.")
    parser.add_argument("--env_pattern", type=str, help="env run pattern sample / adv.")
    parser.add_argument("--save_path", type=str, help="episode save path.")
    parser.add_argument("--sample_num", type=str, help="the sample num.")
    parser.add_argument("--template", type=str, help="use model template.")


    args = parser.parse_args()
    return args


def main(args):
    # read dataset
    if os.path.exists(args.dataset_path):
        dataset = untils.read_data(args.dataset_path)
        dataset = dataset
        print(f"Loaded {len(dataset)} data!")
    else:
        raise Exception(f"{args.dataset_path} is not exist!")

    # read tools
    if os.path.exists(args.tool_path):
        tools = untils.read_data(args.tool_path)
        print(f"Loaded {len(tools)} tool!")
        tool_dict = untils.get_tool_dict(tools)
    else:
        raise Exception(f"{args.tool_path} is not exist!")



    if args.run_pattern == "sample":
        if args.agent_model_name == "llama":
            sample.run_sample_llama(
                dataset=dataset,
                tools=tools,
                tool_dict=tool_dict,
                model_path=args.model_path,
                adapter_model_path=args.adapter_model_path,
                log_save_path=args.log_save_path,
                temperature=args.temperature,
                sample_num=args.sample_num,
                save_path=args.save_path,
                agent_pattern="sample",
            )
        else:
            sample.run_sample_api(
                dataset=dataset,
                tools=tools,
                tool_dict=tool_dict,
                agent_model_name=args.agent_model_name,
                log_save_path=args.log_save_path,
                temperature=args.temperature,
                sample_num=args.sample_num,
                save_path=args.save_path,
                agent_pattern="sample",
            )
    elif args.run_pattern == "expert":
        if args.agent_model_name == "llama":
            expert.run_sample_llama(
                dataset=dataset,
                tools=tools,
                tool_dict=tool_dict,
                model_path=args.model_path,
                env_model_name=args.env_model_name,
                reflector_model_name=args.reflector_model_name,
                evaluation_model_path="/data/zxy/llama3.1_instruction/",
                evaluation_adapter_model_path="/data/zxy/models/safety/",
                temperature=args.temperature,
                sample_num=args.sample_num,
                save_path=args.save_path,
                agent_pattern="sample",
            )
        else:
            expert.run_sample_api(
                dataset=dataset,
                tools=tools,
                tool_dict=tool_dict,
                model_name=args.agent_model_name,
                env_model_name=args.env_model_name,
                reflector_model_name=args.reflector_model_name,
                expert_model_name=args.expert_model_name,
                evaluation_model_path="/data/zxy/llama3.1_instruction/",
                evaluation_adapter_model_path="/data/zxy/models/safety/",
                temperature=args.temperature,
                sample_num=args.sample_num,
                save_path=args.save_path,
            )
    elif args.run_pattern == "test":
        test.run_sample_llama(
            dataset=dataset,
            tools=tools,
            tool_dict=tool_dict,
            model_name=args.agent_model_name,
            model_path=args.model_path,
            adapter_model_path=args.adapter_model_path,
            temperature=args.temperature,
            sample_num=args.sample_num,
            save_path=args.save_path,
            template=args.template,
        )
    else:
        safety_sample.run_sample_api(
            dataset=dataset,
            tools=tools,
            tool_dict=tool_dict,
            agent_model_name="qwen-plus",
            log_save_path=args.log_save_path,
            temperature=0.5,
            sample_num="0",
            save_path="/data/zxy/save/expert/",
            agent_pattern="safety",
        )

        # safety_sample.run_sample_llama(
        #     dataset=dataset,
        #     tools=tools,
        #     tool_dict=tool_dict,
        #     model_path="/data/zxy/Llama-3.1-70B-Instruct",
        #     log_save_path=args.log_save_path,
        #     temperature=0.5,
        #     sample_num="0",
        #     save_path="/data/zxy/save/expert/",
        #     agent_pattern="sample",
        # )

    return


if __name__ == '__main__':
    args = get_args()
    main(args)