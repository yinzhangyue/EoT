# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2023/9/10
import re
from collections import Counter
import numpy as np
from typing import Union, Any
from math import isclose
import jsonlines
from ipdb import set_trace


class Metric:
    def __init__(self) -> None:
        pass

    def most_common(self, lst):
        assert lst != [], "The list is empty!"
        new_lst = [i for i in lst if i != ""]
        return Counter(new_lst).most_common(1)[0][0]

    def get_consistency(self, response_list: list):
        lst = self.process_pred_list(response_list)
        assert lst != [], "The list is empty!"
        new_lst = [_ for _ in lst if _ != ""]
        return Counter(new_lst).most_common(1)[0][1]

    def process_pred(self, response: str) -> str:
        return response

    def process_pred_list(self, response_list: list) -> list:
        pred_list = []
        for response in response_list:
            pred = self.process_pred(response)
            pred_list.append(pred)
        return pred_list

    def cal_acc(self, pred: str, answer: str) -> int:
        return 1 if pred == answer else 0

    def get_acc(self, response_list: list, answer: str):
        pred = self.most_common(self.process_pred_list(response_list))
        return self.cal_acc(pred, answer)


# Math Reasoning
class GSM8K_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        preds = response.split("the answer is")
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if pred == []:
            pred = ""
        else:
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]

            if pred[-1] == ".":
                pred = pred[:-1]
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return None
        return 1 if np.array([float(pred)]) == np.array([float(answer.replace(",", ""))]) else 0


class MultiArith_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class SingleEq_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class AddSub_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class AQuA_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        preds = response.split("the answer is")
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]
        pred = pred.replace(",", "")
        pred = re.findall(r"A|B|C|D|E", pred)

        if pred == []:
            pred = ""
        elif answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return None
        return 1 if pred.lower() == answer.replace(".", "").lower() else 0


class SVAMP_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


# Commonsense Reasoning
class CSQA_Metric(AQuA_Metric):
    def __init__(self) -> None:
        super().__init__()


class StrategyQA_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        preds = response.split("the answer is")
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]
        pred = pred.lower()
        pred = re.sub("\"|'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]

        if pred == []:
            pred = ""
        elif answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return None
        return 1 if pred == answer else 0
