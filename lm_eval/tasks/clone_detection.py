"""
CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Clone Detection (BCB) fron CoedXGLUE
Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others
The task is binary classification
"""
from lm_eval.base import Task
import re
import evaluate
import json

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@inproceedings{svajlenko2014towards,
  title={Towards a big data curated benchmark of inter-project code clones},
  author={Svajlenko, Jeffrey and Islam, Judith F and Keivanloo, Iman and Roy, Chanchal K and Mia, Mohammad Mamun},
  booktitle={2014 IEEE International Conference on Software Maintenance and Evolution},
  pages={476--480},
  year={2014},
  organization={IEEE}
}

@inproceedings{wang2020detecting,
  title={Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree},
  author={Wang, Wenhan and Li, Ge and Ma, Bo and Xia, Xin and Jin, Zhi},
  booktitle={2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  pages={261--271},
  year={2020},
  organization={IEEE}
}
"""


class CloneDetection(Task):
    DATASET_PATH = "code_x_glue_cc_clone_detection_big_clone_bench"
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        instruction = "Answer 'yes' if Code1 and Code2 has semantic equivalent or 'no' otherwise:"
        code = f"\n### Code1:{doc['func1']}\
                 \n### Code2:{doc['func2']}\
                 \n### Answer: "
        prompt = prefix+instruction+code
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return str(int(doc['label']))

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
        #generation = generation.lower()
        yes_word_count = len(re.findall(r'\byes\b', generation)) 
        no_word_count = len(re.findall(r'\bno\b', generation)) 
        if no_word_count>=yes_word_count:
            prediction = "1" #"different semantic"
        else:
            prediction = "0" #"same semantic"
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