# Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication
![License](https://img.shields.io/badge/License-Apache%20License%202.0-green)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

EMNLP 2023: [Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://arxiv.org/abs/2312.01823)


## Introduction 📝

This repository contains the code and data related to the paper "[Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://arxiv.org/pdf/2312.01823.pdf)". In this paper, we introduce a novel cross-model communication framework that incorporates solutions from other models as external insights in problem-solving. Drawing from network topology, we propose four types of communication paradigms: Memory, Report, Relay, and Debate. We also present a confidence evaluation to mitigate the influence of unreliable ideas during the problem-solving process.
![Cover](figures/cover.png)


## Quick Links 🔗

- [Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](#exchange-of-thought-enhancing-large-language-model-capabilities-through-cross-model-communication)
  - [Introduction 📝](#introduction-)
  - [Quick Links 🔗](#quick-links-)
  - [Communication ✨](#communication-)
  - [Requirements 📚](#requirements-)
  - [Quick Start 🚀](#quick-start-)
  - [Bug or Questions? 🤔](#bug-or-questions-)
  - [Citation 📖](#citation-)


## Communication ✨
Drawing inspiration from the interactive discussions among high school students in school, we have designed our models to emulate three high school students: Kitty, Ben, and Peter. Kitty is meticulous and earnest, Ben is smart and quick-witted, and Peter is rich in creativity. During the problem-solving process, they utilize their unique talents to think from different perspectives, subsequently exchanging their ideas to enhance and complement each other's problem-solving strategies.
![Communication](figures/communication.png)


## Requirements 📚
Please make sure you have the following requirements installed:
- openai
- torch
- backoff
- tenacity
- transformers
- sentencepiece
- jsonlines
- tqdm
- matplotlib
- ipdb


## Quick Start 🚀
We are currently organizing our code and the prompts, which will be released in the near future. Stay tuned! 🌹


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
