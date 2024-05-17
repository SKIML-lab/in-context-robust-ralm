import regex, unicodedata, argparse
from datasets import Dataset
from typing import Callable
from collections import Counter
import os, string, re
## From Contriever repo : https://github.com/facebookresearch/contriever/blob/main/src/evaluation.py#L23
def extract_model_name(text: str) -> str:
    if "gpt" in text.lower():
        if "inst" in text.lower():
            return "GPT-INST"
        else:
            return "GPT"
    elif "llama" in text.lower():
        if "70b" in text.lower():
            return "LLAMA-70B"
        else:
            return "LLAMA-8B"
    elif "qwen" in text.lower():
        if "110b" in text.lower():
            return "Qwen-110B"
        elif "72b" in text.lower():
            return "Qwen-72B"
        elif "14b" in text.lower():
            return "Qwen-14B"
        else:
            return "Qwen-7B"

def extract_dataset_name(text: str) -> str:
    if "nq" in text.lower():
        return "NQ"
    elif "tqa" in text.lower():
        return "TQA"
    elif "webq" in text.lower():
        return "WebQ"
    elif "popqa" in text.lower():
        return "PopQA"
    else:
        return "Unknown"

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_run_name(args):
    model_name = extract_model_name(args.llm)
    if args.dummy_data:
        model_name = "*DUMMY*" + model_name
    run_name = model_name+ "||"+ f"{extract_dataset_name(args.dataset)}||{args.demon_order}||"
    if args.filter_uncertain:
        run_name += "filtered-"
    if args.task == "unanswerable":
        run_name += "unans-"
    elif args.task == "conflict":
        run_name += "conf-"
    run_name += f"R:{args.case_retrieval}||"
    run_name += f"Q:{args.qa_demon_size}-{args.unans_demon_method}U:{args.unans_demon_size}-{args.conflict_demon_method}C:{args.conflict_demon_size}"
    if args.prefix:
        run_name = args.prefix + "-" + run_name
    return run_name

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def has_answer(answers: list[str], text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)
    if isinstance(answers, dict):
        answers = answers["text"]
    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def filter_and_map(dataset: Dataset,
                   filter_fn: Callable,
                   map_fn: Callable,
                   out_col: str,
                   num_proc: int = min(16, os.cpu_count())) -> Dataset:
    result = []
    dataset = dataset.add_column("temp_id", range(len(dataset)))
    sub_dataset = dataset.filter(filter_fn, num_proc=os.cpu_count())
    sub_dataset = sub_dataset.map(map_fn, num_proc=num_proc)
    for i in range(len(dataset)):
        if i not in sub_dataset["temp_id"]:
            result.append(None)
        else:
            idx = sub_dataset["temp_id"].index(i)
            result.append(sub_dataset[out_col][idx])
    if out_col in dataset.column_names:
        dataset = dataset.remove_columns(out_col)
    dataset = dataset.add_column(out_col, result)
    dataset = dataset.remove_columns("temp_id")
    return dataset

def validate_args(args) -> bool:
    pass

def normalize_answer(s: str):
    if not s:
        return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em(prediction, ground_truth, normalize_fn: Callable):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

def f1(prediction, ground_truth, normalize_fn: Callable):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truths):
    return max([f1(prediction, gt, normalize_answer) for gt in ground_truths])

def exact_match_score(prediction, ground_truths):
    return max([em(prediction, gt, normalize_answer) for gt in ground_truths])