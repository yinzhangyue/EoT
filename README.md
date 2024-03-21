# Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication

![License](https://img.shields.io/badge/License-Apache%20License%202.0-green)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

EMNLP 2023: [Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://aclanthology.org/2023.emnlp-main.936/)


## Introduction 📝

This repository contains the code and data related to the paper "[Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://arxiv.org/pdf/2312.01823.pdf)". In this paper, we introduce a novel cross-model communication framework that incorporates solutions from other models as external insights in problem-solving. Drawing from network topology, we propose four types of communication paradigms: Memory, Report, Relay, and Debate. We also present a confidence evaluation to mitigate the influence of unreliable ideas during the problem-solving process.

![Cover](figures/cover.png)


## Quick Links 🔗

- [Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](#exchange-of-thought-enhancing-large-language-model-capabilities-through-cross-model-communication)
  - [Introduction 📝](#introduction-)
  - [Quick Links 🔗](#quick-links-)
  - [Communication ✨](#communication-)
  - [Requirements 📚](#requirements-)
  - [Data 💾](#data-)
  - [Quick Start 🚀](#quick-start-)
    - [Tips](#tips)
  - [Evaluation 💻](#evaluation-)
  - [Bug or Questions? 🤔](#bug-or-questions-)
  - [Citation 📖](#citation-)


## Communication ✨

Drawing inspiration from the interactive discussions among high school students in school, we have designed our models to emulate three high school students: Kitty, Ben, and Peter. Kitty is meticulous and earnest, Ben is smart and quick-witted, and Peter is rich in creativity. During the problem-solving process, they utilize their unique talents to think from different perspectives, subsequently exchanging their ideas to enhance and complement each other's problem-solving strategies.
![Communication](figures/communication.png)


## Requirements 📚

Please make sure you have the following requirements installed:
- openai > 1.14.0
- backoff
- tenacity
- jsonlines
- ipdb


## Data 💾

Our dataset originates from [Large Language Models are Zero-Shot Reasoners](https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset), generously shared by Takeshi Kojima. We employ prompts from [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf) to guide models in generating initial reasoning processes.

Considering the costly overhead of API calls, we provide the initial reasoning outcomes saved in the `jsonl` format within the `samples` folder. You can directly load example data from the `samples` folder to run Exchange-of-Thought (EoT). The complete dataset is stored at [Google Drive](https://drive.google.com/drive/u/0/folders/1iSCD_459LmJXRq3tq-c3BnYaLjmt-9wi).


## Quick Start 🚀

We provide a reference startup script in [main.py](code/main.py). Below is a comprehensive explanation of the arguments for launching the EoT script, allowing for customized training runs.

To run the GSM8K dataset with the Memory communication mode, use the following script:

```bash
python eot.py --task GSM8K --data-path [Initital_Data_Path] --record-path [Your_Record_Path] --communication-mode Memory --inference-model gpt-35-turbo-0301
```
You'll need to replace `[Initital_Data_Path]` and `[Your_Record_Path]` with your data directory and result storage directory, respectively. Initial data can be referred to in the Data section.

When running `eot.py`, you can customize EoT with the following input arguments:
- `task`: The reasoning task EoT needs to complete, e.g., GSM8K, AQuA, CSQA, etc.
- `data-path`: The storage path for initial reasoning outcomes, in `jsonl` format, requiring "question", "answer", and "response_list" fields.
- `record-path`: The storage location for output results.
- `communication-mode`: The communication method for EoT, options include "Memory", "Report", "Relay", "Debate".
- `inference-model`: The model used for communication, default is gpt-35-turbo-0301. We support all OpenAI models for EoT.
- `communicator-num`: The number of participants in communication, default is 3.
- `max-round`: The maximum number of communication rounds, default is 5, after which communication ends.

### Tips

We support APIs from OpenAI and Azure. We recommend using Azure's API to avoid potential instability with OpenAI's API.

Currently, we only support 3 communicators. You might need to modify the content in [eot.py](code/eot.py) to accommodate more participants.

You can adapt EoT to new reasoning tasks or scenarios, or DIY communication modes, by modifying the `exchange` function in `eot.py`. By default, EoT doesn't require additional example samples during the communication process, relying solely on the models themselves for interaction and verification. In complex scenarios, we recommend providing a suitable example sample to guide models in better understanding each other's intentions and completing communication.

You can construct your desired character information for communication in [prompt.py](code/prompt.py). For simplicity, we use a unified System Prompt. However, we've found that for hard tasks, using specially designed System Prompts can further improve model performance.


## Evaluation 💻

In [metric.py](code/metric.py), we provide evaluation methods for various tasks, utilizing accuracy to assess the model's reasoning performance. We offer the `process_pred` function for evaluating single reasoning chain and the `process_pred_list` function for evaluating multiple reasoning chains.

To facilitate future research and analysis, we have made the communication results of EoT available at [Google Drive](https://drive.google.com/drive/u/0/folders/1ehjsytg8RlvRJ8TsDkWfTwCceJ4Xcm1B)🤗.


## Bug or Questions? 🤔

If you have any suggestions or questions, feel free to email us at yinzhangyue@126.com. If you encounter any issues while using the code, or if you find any bugs, please open a new issue on GitHub. This is a preliminary work and we are very much open to any constructive feedback that could help us improve. Thank you for your attention!


## Citation 📖

If you are interested in our work, please use the following citation format when referencing our paper:
```bibtex
@inproceedings{yin2023exchange,
    title     = "Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication",
    author    = "Yin, Zhangyue  and
      Sun, Qiushi  and
      Chang, Cheng  and
      Guo, Qipeng  and
      Dai, Junqi  and
      Huang, Xuanjing  and
      Qiu, Xipeng",
    editor    = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month     = dec,
    year      = "2023",
    address   = "Singapore",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2023.emnlp-main.936",
    pages     = "15135--15153",
}
```
