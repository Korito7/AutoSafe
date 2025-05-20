from openai import OpenAI
import time
from transformers import pipeline
import torch
from transformers import AutoTokenizer
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from pydantic import BaseModel, Field
import nest_asyncio
nest_asyncio.apply()

class Api():
    def __init__(self, base_url, api_key, formatting_fields=None):
        self.base_url = base_url
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,  # ModelScope Token
            timeout=18000,
        )

        self.formatting_fields = formatting_fields

        if formatting_fields:
            self.formatting_model = OpenAIModel(
            "Qwen/Qwen2.5-32B-Instruct",
            base_url="https://api-inference.modelscope.cn/v1/",
            api_key="625c1b30-4492-4fb0-8bad-a12278ce6b7f",)

            if formatting_fields == "gen_case":
                self.formatting_class = self.create_research_class_in_gen_case()
            elif formatting_fields == "agent":
                self.formatting_class = self.create_research_class_in_agent()
            elif formatting_fields == "observation":
                self.formatting_class = self.create_research_class_in_observation()

            self.formatting_agent = Agent(self.formatting_model,
                                          result_type=self.formatting_class,
                                          system_prompt=(
                                          "You are a helpful formatting assistant and you never give your opinion.",
                                          "You just received a given input and transform or summary it into a structured results to return.",
                                          "All outputs should be original words from the input. You should not summarize or restate them using synonyms.")
                                          )

    def create_research_class_in_gen_case(self):
        class ResearchResult(BaseModel):
            instruction: str = Field(
                description="One sentence is used to describe a task instruction that the user asks the Agent to execute.")
            expected_achievement: str = Field(
                description="One sentence to overview how the user wants the Agent to execute instructions to achieve the goal.")
            risky_outcomes: str = Field(
                description="One sentence to overview the unsafe actions that may occur during Agent execution.")

        return ResearchResult

    def create_research_class_in_agent(self):
        class ResearchResult(BaseModel):
            thought: str = Field(
                description="Thought part: One sentence describes the thinking process of the Agent.")
            action: str = Field(
                description="Action part: A tool name Agent chooses to use. Must use the original words from the corresponding parts in the input. If 'Final Answer' in content, this Action is Final Answer.")
            action_input: str = Field(
                description="Action Input part: Input parameters corresponding to the tool. Must use the original words from the corresponding parts in the input.")

        return ResearchResult


    def create_research_class_in_observation(self):
        class ResearchResult(BaseModel):
            observation: str = Field(
                description="Observation part: A simulation of the results after the tool is executed.")

        return ResearchResult


    def _call_api(self, model_name, messages, temperature, retries=5):
        """
        封装 API 调用，带有异常处理和重试机制。
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,  # ModelScope Model-Id
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                )
                return response
            except Exception as e:
                print(f"API 调用失败: {e}, 正在重试 ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)  # 指数退避等待

        # 5 次重试仍然失败，抛出异常并终止程序
        raise RuntimeError("API 调用失败，已达到最大重试次数")


    async def _formatting(self, results, retries=5):
        for attempt in range(retries):
            try:
                formatted_result = await self.formatting_agent.run(results.strip())
                if self.formatting_fields == "gen_case":
                    instruction = formatted_result.data.instruction
                    expected_achievement = formatted_result.data.expected_achievement
                    risky_outcomes = formatted_result.data.risky_outcomes
                    return instruction, expected_achievement, risky_outcomes
                elif self.formatting_fields == "agent":
                    thought = formatted_result.data.thought
                    action = formatted_result.data.action
                    action_input = formatted_result.data.action_input
                    return thought, action, action_input
                elif self.formatting_fields == "observation":
                    observation = formatted_result.data.observation
                    return observation

            except Exception as e:
                print(f"API 调用失败: {e}, 正在重试 ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)  # 指数退避等待
        # 5 次重试仍然失败，抛出异常并终止程序
        try:
            if self.formatting_fields == "gen_case":
                instruction = results.split("[User Instruction]:")[1].split("[Expected Achievements]")[0].strip()
                expected_achievement = results.split("[Expected Achievements]:")[1].split("[Potential Risky Outcomes]")[0].strip()
                risky_outcomes = results.split("[Potential Risky Outcomes]:")[1].strip()

                return instruction, expected_achievement, risky_outcomes
            elif self.formatting_fields == "agent":
                assert 0 == 1


        except:
            raise RuntimeError("API 调用失败，已达到最大重试次数")


    async def get_llm_response(self, model_name, messages, temperature=0.3):
        response = self._call_api(model_name, messages, temperature)

        try:
            result = response.choices[0].message.content
            cost_tokens = dict(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            if self.formatting_fields == "gen_case":
                instruction, expected_achievement, risky_outcomes = await self._formatting(result)
                return instruction, expected_achievement, risky_outcomes, cost_tokens
            elif self.formatting_fields == "agent":
                result = result.split("Observation")[0].strip()
                thought, action, action_input = await self._formatting(result)
                if "action" in action:
                    try:
                        action = action.split(":")[1].strip()
                    except:
                        action = action
                return thought, action, action_input, cost_tokens
            elif self.formatting_fields == "observation":
                # observation = await self._formatting(result)
                observation = result.strip()
                return observation, cost_tokens

        except AttributeError:
            raise RuntimeError("API 返回数据格式异常")

        return result, cost_tokens


class Local():
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def _call_pipeline(self, messages, temperature, retries=5):
        """
        封装 pipeline 调用，带有异常处理和重试机制。
        """
        for attempt in range(retries):
            try:
                generation_args = {
                    "temperature": temperature,
                    "do_sample": False,
                    "max_new_tokens": 1024,
                }

                return self.pipeline(messages, **generation_args)
            except Exception as e:
                print(f"文本生成失败: {e}, 正在重试 ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)  # 指数退避等待

        raise RuntimeError("文本生成失败，已达到最大重试次数")

    def get_llm_response(self, messages, temperature=0.0):
        output = self._call_pipeline(messages, temperature)

        try:
            result = output[0]["generated_text"][2]["content"].strip()
        except (IndexError, KeyError, AttributeError):
            raise RuntimeError("模型输出格式异常")

        # 计算输入token数量
        input_text = " ".join([message['content'] for message in messages])
        tokens = self.tokenizer(input_text, return_tensors="pt")
        input_tokens = tokens.input_ids.shape[1]

        output_text = result
        tokens = self.tokenizer(output_text, return_tensors="pt")
        output_tokens = tokens.input_ids.shape[1]

        total_tokens = input_tokens + output_tokens

        cost_tokens = dict(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return result, cost_tokens