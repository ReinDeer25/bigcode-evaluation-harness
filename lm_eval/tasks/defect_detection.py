"""
CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Defect Detection fron CoedXGLUE
Given a source code, the task is to identify whether it is an insecure code that may attack software systems, 
such as resource leaks, use-after-free vulnerabilities and DoS attack. 
The task is binary classification
"""
from lm_eval.base import Task
import re
import evaluate
import json

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@inproceedings{zhou2019devign,
  title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
  author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10197--10207},
  year={2019}
}
"""


class DefectDetection(Task):
    DATASET_PATH = "code_x_glue_cc_defect_detection"
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instruction = '''Is there a defect in the Code, and respond to YES or NO.'''
        code = doc['func']
        prompt = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{code}

### Response:'''
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return str(int(doc['target']))

    def postprocess_generation(self, generation, idx):
        # TODO: define the postprocessing for the LM generation
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        # logic is to count the word from generation text and used as the final prediction
        secure_word_count = len(re.findall(r'\bsecure\b', generation)) 
        insecure_word_count = len(re.findall(r'\binsecure\b', generation)) 
        if insecure_word_count>=secure_word_count:
            prediction = "1" #"insecure"
        else:
            prediction = "0" #"secure"
        return generation

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        accuracy_metric = evaluate.load("accuracy")
        recall_metric = evaluate.load("recall")
        precision_metric = evaluate.load("precision")
        f1_metric = evaluate.load("f1")
        preds = [gen[0] for gen in generations]
        return  {
            "Accuracy": accuracy_metric.compute(predictions=preds, references=references),
            "Recall":recall_metric.compute(predictions=preds, references=references), 
            "Precision":precision_metric.compute(predictions=preds, references=references), 
            "F1":f1_metric.compute(predictions=preds, references=references)
        }