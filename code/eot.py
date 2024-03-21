# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/3/10
import jsonlines
from ipdb import set_trace
from collections import Counter
from prompt import System_Prompt
from inference import Inference_Model
from metric import GSM8K_Metric, MultiArith_Metric, SingleEq_Metric, AddSub_Metric, AQuA_Metric, SVAMP_Metric, CSQA_Metric, StrategyQA_Metric
import argparse
import os

parser = argparse.ArgumentParser()

# Basic Setting
parser.add_argument("--task", type=str, required=True, help="Reasoning Task")
parser.add_argument("--data-path", type=str, required=True, help="path to the data file")
parser.add_argument("--record-path", type=str, required=True, help="path to save the record file")
parser.add_argument("--communication-mode", default="Memory", choices=["Memory", "Report", "Relay", "Debate"], type=str, help="communication mode for EoT")
parser.add_argument("--inference-model", default="gpt-35-turbo-0301", choices=["gpt-35-turbo-0301", "gpt-4-0314"], type=str, help="inferece model for EoT")
# Hyperparameters
parser.add_argument("--communicator-num", default=3, type=int, help="number of communicators")
parser.add_argument("--max-round", default=5, type=int, help="maximum number of communication rounds")

args = parser.parse_args()


def read_jsonl_file(file_path: str):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


class EOT:
    def __init__(self, system_prompt: dict, metric, max_round: int, communicator_num: int, communication_mode: str, inference_model, data_path: str, record_path: str, format_hint: str = "", **kwargs):
        self.system_prompt = system_prompt
        self.metric = metric
        self.max_round = max_round
        self.communicator = list(self.system_prompt.keys())
        self.communicator_num = communicator_num
        self.communication_mode = communication_mode
        self.inference_model = inference_model
        self.data_path = data_path
        self.record_path = record_path
        self.format_hint = format_hint

    def construct_prompt(self, my_solution: str, participant1: str, response1: str, confidence1: int = -1, participant2: str = "", response2: str = "", confidence2: int = -1):
        query = """Please consider the example provided and think it step by step. \nQuestion: {}""".format(self.question)

        if confidence1 == -1 and participant2 == "":
            query = """Please consider the example provided and think it step by step.
            Question: {}
            Your Solution: {}
            Here is a solution process from your friend:
            {}'s Solution: {}
            Based on your friend's solution, carefully re-examine your previous answer. Utilize your talent and critical thinking to provide a new step-by-step solution process.
            Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, "the answer is{}."
            """.format(
                self.question, my_solution, participant1, response1, self.format_hint
            )

        elif confidence1 == -1 and participant2 != "":
            query = """Please consider the example provided and think it step by step.
            Question: {}
            Your Solution: {}
            Here is a solution process from your friend:
            {}'s Solution: {}
            {}'s Solution: {}
            Based on your friend's solution, carefully re-examine your previous answer. Utilize your talent and critical thinking to provide a new step-by-step solution process.
            Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, "the answer is{}."
            """.format(
                self.question, my_solution, participant1, response1, participant2, response2, self.format_hint
            )

        elif confidence1 != -1 and participant2 == "":
            query = """Please consider the example provided and think it step by step.
            Question: {}
            Your Solution: {}
            Here is a solution process from your friend:
            {}'s Solution: {}
            {}'s confidence in this solution is: {}
            Based on your friend's solution, carefully re-examine your previous answer. If your friend's confidence level is below 0.5, it suggests a high probability that the solution might be incorrect. Remember, solutions with high confidence can also be wrong. Utilize your talent and critical thinking to provide a new step-by-step solution process.
            Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, "the answer is{}."
            """.format(
                self.question, my_solution, participant1, response1, participant1, confidence1, self.format_hint
            )

        elif confidence1 != -1 and participant2 != "":

            query = """Please consider the example provided and think it step by step.
            Question: {}
            Your Solution: {}
            Here is a solution process from your friend:
            {}'s Solution: {}
            {}'s confidence in this solution is: {}
            {}'s Solution: {}
            {}'s confidence in this solution is: {}
            Based on your friend's solution, carefully re-examine your previous answer. If your friend's confidence level is below 0.5, it suggests a high probability that the solution might be incorrect. Remember, solutions with high confidence can also be wrong. Utilize your talent and critical thinking to provide a new step-by-step solution process.
            Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, "the answer is{}."
            """.format(
                self.question, my_solution, participant1, response1, participant1, confidence1, participant2, response2, participant2, confidence2, self.format_hint
            )

        return query

    def exchange(self, communication_round=1):
        query_dict = {}
        # 0->A 1->B 2->C
        if communication_round <= 1:
            if self.communication_mode == "Memory":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], participant2=self.communicator[1], response2=self.message_record_dict[self.communicator[1]][-1])
            elif self.communication_mode == "Report":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1])
            elif self.communication_mode == "Relay":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[2], response1=self.message_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1])
            elif self.communication_mode == "Debate":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[2], response1=self.message_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1])
        else:
            if self.communication_mode == "Memory":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], confidence1=self.confidence_record_dict[self.communicator[1]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1], confidence2=self.confidence_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], confidence1=self.confidence_record_dict[self.communicator[0]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1], confidence2=self.confidence_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], confidence1=self.confidence_record_dict[self.communicator[0]][-1], participant2=self.communicator[1], response2=self.message_record_dict[self.communicator[1]][-1], confidence2=self.confidence_record_dict[self.communicator[1]][-1])
            elif self.communication_mode == "Report":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], confidence1=self.confidence_record_dict[self.communicator[1]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1], confidence2=self.confidence_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], confidence1=self.confidence_record_dict[self.communicator[0]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], confidence1=self.confidence_record_dict[self.communicator[0]][-1])
            elif self.communication_mode == "Relay":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], confidence1=self.confidence_record_dict[self.communicator[1]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[2], response1=self.message_record_dict[self.communicator[2]][-1], confidence1=self.confidence_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[0], response1=self.message_record_dict[self.communicator[0]][-1], confidence1=self.confidence_record_dict[self.communicator[0]][-1])
            elif self.communication_mode == "Debate":
                query_dict[self.communicator[0]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[0]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], confidence1=self.confidence_record_dict[self.communicator[1]][-1], participant2=self.communicator[2], response2=self.message_record_dict[self.communicator[2]][-1], confidence2=self.confidence_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[1]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[1]][-1], participant1=self.communicator[2], response1=self.message_record_dict[self.communicator[2]][-1], confidence1=self.confidence_record_dict[self.communicator[2]][-1])
                query_dict[self.communicator[2]] = self.construct_prompt(my_solution=self.message_record_dict[self.communicator[2]][-1], participant1=self.communicator[1], response1=self.message_record_dict[self.communicator[1]][-1], confidence1=self.confidence_record_dict[self.communicator[1]][-1])
        return query_dict

    def cal_confidence(self, participant):
        arr = self.pred_record_dict[participant]
        if len(arr) > 1:
            # Count the occurrences of each item
            item_counts = Counter(arr)
            # Find the highest number of occurrences
            max_occurrences = max(item_counts.values())
            # Calculate the confidence
            confidence = max_occurrences / len(arr)
            self.confidence_record_dict[participant].append(confidence)

    def run(self):
        data_list = read_jsonl_file(self.data_path)

        total_acc = 0
        total_num = len(data_list)
        for data in data_list:
            self.question = data["question"]
            self.answer = data["answer"]
            self.response_list = data["response_list"]
            self.message_record_dict = {}
            self.pred_record_dict = {}
            self.confidence_record_dict = {}
            # Reach Agreement
            if self.metric.get_consistency(self.response_list) == self.communicator_num:
                self.pred = self.metric.process_pred_list(self.response_list)
                self.acc = self.metric.get_acc(self.response_list, self.answer)
                total_acc += self.acc
                self.save_record(data)
                continue

            for i in range(self.communicator_num):
                self.message_record_dict[self.communicator[i]] = [self.response_list[i].replace("\n", " ")]
                self.pred_record_dict[self.communicator[i]] = [self.metric.process_pred(self.response_list[i])]
                self.confidence_record_dict[self.communicator[i]] = []
            for communication_round in range(1, self.max_round + 1):
                query_dict = self.exchange(communication_round)
                for participant in self.communicator:
                    participant_response = self.inference_model.get_info(query=query_dict[participant], System_Prompt=self.system_prompt[participant]).replace("\n", " ")
                    # Debug
                    # print("Question: ", self.question, "\nAnswer: ", self.answer, "\nParticipant: ", participant, "\nQuery:", query_dict[participant], "\nResponse: ", participant_response)
                    self.message_record_dict[participant].append(participant_response)
                    self.pred_record_dict[participant].append(self.metric.process_pred(participant_response))
                    self.cal_confidence(participant)
                # Reach Agreement
                last_pred_list = [self.pred_record_dict[participant][-1] for participant in self.communicator]
                if self.metric.get_consistency(last_pred_list) == self.communicator_num:
                    self.pred = self.metric.process_pred_list(last_pred_list)
                    self.acc = self.metric.get_acc(last_pred_list, self.answer)
                    total_acc += self.acc
                    self.save_record(data)
                    break
            # Reach the maximum number of communication rounds
            if self.metric.get_consistency(last_pred_list) != self.communicator_num:
                self.pred = self.metric.process_pred_list(last_pred_list)
                self.acc = self.metric.get_acc(last_pred_list, self.answer)
                total_acc += self.acc
                self.save_record(data)
        print("Total Accuracy: ", total_acc / total_num)

    def save_record(self, data, record_filename=""):
        data["message_record_dict"] = self.message_record_dict
        data["pred_record_dict"] = self.pred_record_dict
        data["confidence_record_dict"] = self.confidence_record_dict
        data["acc"] = self.acc
        data["pred"] = self.pred
        with jsonlines.open(self.record_path, "a") as writer:
            writer.write(data)
        if record_filename != "":
            with jsonlines.open(record_filename, "a") as writer:
                writer.write(data)


if __name__ == "__main__":
    format_hint_dict = {"GSM8K": "", "MultiArith": "", "AddSub": "", "SingleEq": "", "AQuA": " (A/B/C/D/E)", "SVAMP": "", "CSQA": " (A/B/C/D/E)", "StrategyQA": " (yes/no)"}
    metric_dict = {"GSM8K": GSM8K_Metric, "MultiArith": MultiArith_Metric, "AddSub": AddSub_Metric, "SingleEq": SingleEq_Metric, "AQuA": AQuA_Metric, "SVAMP": SVAMP_Metric, "CSQA": CSQA_Metric, "StrategyQA": StrategyQA_Metric}
    task = args.task
    data_path = args.data_path
    record_path = args.record_path
    communication_mode = args.communication_mode
    model = args.inference_model
    inference_model = Inference_Model(model)
    communicator_num = args.communicator_num
    max_round = args.max_round
    system_prompt = System_Prompt

    format_hint = format_hint_dict[task]
    metric = metric_dict[task]()

    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    print("Task:{} Model:{} Communication Mode:{}".format(task, model, communication_mode))
    eot = EOT(system_prompt, metric, max_round, communicator_num, communication_mode, inference_model, data_path, record_path, format_hint)
    eot.run()
