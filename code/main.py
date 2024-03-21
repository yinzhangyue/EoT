# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/3/10
import subprocess


def main():
    # GSM8K Memory
    subprocess.call(
        "python eot.py \
            --task GSM8K \
            --data-path initial_prompts/GSM8K_EoT_gpt-3.5-turbo-0301.jsonl \
            --record-path records/GSM8K_EoT_log_gpt-3.5-turbo-0301.jsonl \
            --communication-mode Memory \
            --inference-model gpt-35-turbo-0301",
        shell=True,
    )
    # StrategyQA Report
    # subprocess.call(
    #     "python eot.py \
    #         --task StrategyQA \
    #         --data-path initial_promptsStrategyQA_EoT_gpt-3.5-turbo-0301.jsonl \
    #         --record-path records/StrategyQA_EoT_log_gpt-3.5-turbo-0301.jsonl \
    #         --communication-mode Report \
    #         --inference-model gpt-35-turbo-0301",
    #     shell=True,
    # )


if __name__ == "__main__":
    main()
