# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/3/10
from openai import OpenAI
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from ipdb import set_trace

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(api_version="Azure_API_VERSION", azure_endpoint="Azure_ENDPOINT", api_key="Azure_API_KEY")

# this is also the default, it can be omitted
# client = OpenAI(api_key="OPENAI_API_KEY")


class Inference_Model:
    def __init__(self, default_model: str) -> None:
        self.model_name = default_model

    def get_info(self, query: str, System_Prompt: str, messages_list: list = []):
        if messages_list == []:
            completion = client.chat.completions.create(
                model=self.model_name,  # engine = "deployment_name".
                messages=[
                    {
                        "role": "system",
                        "content": System_Prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            )
        else:
            completion = client.chat.completions.create(model=self.model_name, messages=messages_list)
        return completion.choices[0].message.content

    def get_info_openai(self, query: str, System_Prompt: str, messages_list: list = []):
        if messages_list == []:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": System_Prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            )
        else:
            completion = client.chat.completions.create(model=self.model_name, messages=messages_list)
        return completion.choices[0].message.content
