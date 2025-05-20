from openai import OpenAI
import time

class AgentApi():
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,  # ModelScope Token
        )

    def _call_api(self, model_name, messages, temperature=0.3, retries=5):
        """
        封装 API 调用，带有异常处理和重试机制。
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,  # ModelScope Model-Id
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                return response
            except Exception as e:
                print(f"API 调用失败: {e}, 正在重试 ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)  # 指数退避等待

        # 5 次重试仍然失败，抛出异常并终止程序
        raise RuntimeError("API 调用失败，已达到最大重试次数")

    def get_llm_response(self, model_name, messages, temperature=0.3):
        response = self._call_api(model_name, messages, temperature)

        try:
            result = response.choices[0].message.content

            # thought, action, action_input = self._formatting(result)

            cost_tokens = dict(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        except AttributeError:
            raise RuntimeError("API 返回数据格式异常")

        return result, cost_tokens