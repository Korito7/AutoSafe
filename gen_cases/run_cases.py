import re
import json
import os

import copy
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_path)

import model_calling
import tqdm
from Security_agent_ReAct_2 import untils, generation
import prompt

tools_path = "../data/tools/all_toolkits.json"
tools = json.load(open(tools_path))
tool_dict = untils.get_tool_dict(tools)

Gencase = prompt.Case()



execute_model = model_calling.Api(base_url="https://4.0.wokaai.com/v1/",
                                  api_key="sk-ME8MwN5To9RSDKOVxWzSIQAevThCzh04MSr9EFJ4vhB9Qntq",
                                  formatting_fields="gen_case")
model_name = "claude-3-5-sonnet-20241022"

available_toolkits = [{"Primary Toolkits": toolkit_name, "Auxiliary Toolkits": []} for toolkit_name in tool_dict.keys()]
available_toolkits.extend([
  {"Primary Toolkits": "Terminal", "Auxiliary Toolkits": ["TwitterManager"]},
  {"Primary Toolkits": "Terminal", "Auxiliary Toolkits": ["GoogleSearch", "WebBrowser"]},
  {"Primary Toolkits": "Gmail", "Auxiliary Toolkits": ["GoogleCalendar"]},
  {"Primary Toolkits": "GoogleCalendar", "Auxiliary Toolkits": ["Gmail", "Todoist"]},
  {"Primary Toolkits": "BankManager", "Auxiliary Toolkits": ["Venmo", "Amazon"]},
  {"Primary Toolkits": "Amazon", "Auxiliary Toolkits": ["BankManager", "GoogleHome"]},
  {"Primary Toolkits": "Expedia", "Auxiliary Toolkits": ["GoogleMap", "WebBrowser"]},
  {"Primary Toolkits": "TwitterManager", "Auxiliary Toolkits": ["Terminal", "GoogleSearch"]},
  {"Primary Toolkits": "FacebookManager", "Auxiliary Toolkits": ["GoogleHome", "TwitterManager"]},
  {"Primary Toolkits": "Dropbox", "Auxiliary Toolkits": ["WebBrowser", "GoogleSearch"]},
  {"Primary Toolkits": "Teladoc", "Auxiliary Toolkits": ["GoogleSearch", "EmergencyDispatchSystem"]},
  {"Primary Toolkits": "Slack", "Auxiliary Toolkits": ["Gmail", "WebBrowser"]},
  {"Primary Toolkits": "Twilio", "Auxiliary Toolkits": ["Terminal", "TwitterManager"]},
  {"Primary Toolkits": "GoogleHome", "Auxiliary Toolkits": ["Amazon", "Venmo"]},
  {"Primary Toolkits": "Venmo", "Auxiliary Toolkits": ["BankManager", "Gmail"]},
  {"Primary Toolkits": "GoogleMap", "Auxiliary Toolkits": ["Expedia", "WebBrowser"]},
  {"Primary Toolkits": "EthereumManager", "Auxiliary Toolkits": ["Binance", "TDAmeritrade"]},
  {"Primary Toolkits": "GitHub", "Auxiliary Toolkits": ["WebBrowser", "Slack"]},
  {"Primary Toolkits": "Shopify", "Auxiliary Toolkits": ["InventoryManagementSystem", "FedExShipManager"]},
  {"Primary Toolkits": "NortonIdentitySafe", "Auxiliary Toolkits": ["GoogleSearch", "WebBrowser"]},
  {"Primary Toolkits": "EpicFHIR", "Auxiliary Toolkits": ["Teladoc", "GoogleSearch"]},
  {"Primary Toolkits": "TrafficControl", "Auxiliary Toolkits": ["EmergencyDispatchSystem", "GoogleMap"]},
  {"Primary Toolkits": "Terminal", "Auxiliary Toolkits": ["GitHub", "Slack"]},
  {"Primary Toolkits": "TwitterManager", "Auxiliary Toolkits": ["GoogleSearch", "Slack"]},
  {"Primary Toolkits": "WebBrowser", "Auxiliary Toolkits": ["GoogleMap", "TwitterManager"]},
  {"Primary Toolkits": "Gmail", "Auxiliary Toolkits": ["Slack", "Todoist"]},
  {"Primary Toolkits": "Dropbox", "Auxiliary Toolkits": ["Gmail", "GoogleHome"]},
  {"Primary Toolkits": "Venmo", "Auxiliary Toolkits": ["BankManager", "GoogleHome"]},
  {"Primary Toolkits": "Expedia", "Auxiliary Toolkits": ["GoogleMap", "WebBrowser", "Amazon"]},
  {"Primary Toolkits": "GoogleHome", "Auxiliary Toolkits": ["Amazon", "Venmo"]},
  {"Primary Toolkits": "GitHub", "Auxiliary Toolkits": ["Terminal", "Slack"]},
  {"Primary Toolkits": "FacebookManager", "Auxiliary Toolkits": ["TwitterManager", "Slack"]},
  {"Primary Toolkits": "Teladoc", "Auxiliary Toolkits": ["EmergencyDispatchSystem", "Gmail"]},
  {"Primary Toolkits": "Shopify", "Auxiliary Toolkits": ["InventoryManagementSystem", "GoogleSearch"]},
  {"Primary Toolkits": "TDAmeritrade", "Auxiliary Toolkits": ["EthereumManager", "Binance"]},
  {"Primary Toolkits": "Slack", "Auxiliary Toolkits": ["GitHub", "GoogleSearch"]},
  {"Primary Toolkits": "NortonIdentitySafe", "Auxiliary Toolkits": ["WebBrowser", "GoogleSearch"]},
  {"Primary Toolkits": "EmergencyDispatchSystem", "Auxiliary Toolkits": ["GoogleMap", "TrafficControl"]},
  {"Primary Toolkits": "Binance", "Auxiliary Toolkits": ["TDAmeritrade", "EthereumManager"]},
  {"Primary Toolkits": "GoogleMap", "Auxiliary Toolkits": ["TrafficControl", "GoogleSearch"]},
  {"Primary Toolkits": "Slack", "Auxiliary Toolkits": ["Gmail", "TwitterManager"]},
  {"Primary Toolkits": "EthereumManager", "Auxiliary Toolkits": ["TDAmeritrade", "Binance", "GitHub"]},
  {"Primary Toolkits": "TwitterManager", "Auxiliary Toolkits": ["GoogleHome", "Venmo"]},
  {"Primary Toolkits": "FacebookManager", "Auxiliary Toolkits": ["TwitterManager", "GoogleSearch"]},
  {"Primary Toolkits": "DeepfakeGenerator", "Auxiliary Toolkits": ["GoogleSearch", "TwitterManager"]},
  {"Primary Toolkits": "Amazon", "Auxiliary Toolkits": ["Expedia", "GoogleHome"]},
  {"Primary Toolkits": "InventoryManagementSystem", "Auxiliary Toolkits": ["Shopify", "FedExShipManager"]}
]
)

def get_message(prompts):
    messages = [
        {
            "role": "system",
            "content": prompts[0]
        },
        {
            "role": "user",
            "content": prompts[1]
        }
    ]

    return messages

temperature = 0.3
save_path = "/data/zxy/save/cases/"
case_num = 0
for available_toolkit in available_toolkits:
    for risky in tqdm.tqdm(untils.RISKY_TYPES):
        print("\n\nCurrent available primary toolkits: ", available_toolkit["Primary Toolkits"])
        print("Current available auxiliary toolkits: ", "; ".join(available_toolkit["Auxiliary Toolkits"]))
        print("Current risky type: ", risky)
        print("\n")
        primary_toolkit_name = available_toolkit["Primary Toolkits"]
        primary_toolkit_indexs = [tool_dict[primary_toolkit_name]]
        primary_toolkits = [tools[toolkit_index] for toolkit_index in primary_toolkit_indexs]

        primary_toolkit_strings = "\n".join(
          [untils.create_tool_description(toolkit, level="none") for toolkit in primary_toolkits]
        )

        auxiliary_toolkit_names = available_toolkit["Auxiliary Toolkits"]
        auxiliary_toolkit_indexs = [tool_dict[toolkit_name] for toolkit_name in auxiliary_toolkit_names]
        auxiliary_toolkits = [tools[toolkit_index] for toolkit_index in auxiliary_toolkit_indexs]

        auxiliary_toolkit_strings = "\n".join(
          [untils.create_tool_description(toolkit) for toolkit in auxiliary_toolkits]
        )

        available_toolkits_str = "\n<Primary Toolkits>:\n" + primary_toolkit_strings + "\n\n" + "<Auxiliary Toolkits>:\n" + auxiliary_toolkit_strings

        risky_description = risky + ": " + untils.RISKY_DESCRIPTION[risky]

        gen_case_input_args = {"available_toolkits": available_toolkits_str,
                               "security_risk": risky_description}

        prompts = Gencase.get_prompt(gen_case_input_args)

        messages = get_message(prompts)
        instruction, expected_achievement, risky_outcomes, cost_tokens = execute_model.get_llm_response(model_name, messages, temperature)

        cases = untils.extract_cases(instruction, expected_achievement, risky_outcomes)
        toolkits = copy.deepcopy(auxiliary_toolkit_names)
        toolkits.append(primary_toolkit_name)
        cases["Toolkits"] = toolkits
        cases["Risky"] = risky
        cases["cost_tokens"] = cost_tokens

        case_path = save_path + "case_" + str(case_num) + ".json"
        untils.write_data(cases, case_path)

        case_num += 1