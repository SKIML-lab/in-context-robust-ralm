from datasets import Dataset, load_dataset
from typing import List, Dict
from dataset import dpr_embed
from preprocess import query_masking, normalize_answer
import random, joblib
from random import randint
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
from dataset import build_multiple_indexes
from utils import extract_dataset_name, has_answer, SimpleTokenizer
from case_retrieval import load_case_set

GPT_NORMAL_INST =("Answer the following question based on the provided knowledge. "
                  "Please provide the answer as a single word or term, without forming a complete sentence.\n\n"
                  "{DEMONS}"
                  "Knowledge: {DOCS}\n"
                  "Question: {QUESTION}\n"
                  "Answer:")
GPT_UNANS_INST = ("Use the provided documents to answer questions. Please provide the answer as a single word or term, "
                  "without forming a complete sentence. If the answer cannot be found in the documents, write 'I could not find an answer.'\n\n"
                  "{DEMONS}"
                  "Documents: {DOCS}\n"
                  "Q: {QUESTION}\n"
                  "A:")
GPT_UNANS_INST_V2 = ("Answer the following question based on the provided knowledge. "
                     "If you cannot find the answer in the provided knowledge, please respond with 'unanswerable'. "
                     "Please provide the answer as a single word or term, "
                     "without forming a complete sentence.\n\n"
                    "{DEMONS}"
                    "Knowledge: {DOCS}\n"
                    "Q: {QUESTION}\n"
                    "A:")

GPT_UNANS_INST_V3 = ("Answer the following question based on the provided knowledge. "
                     "If you cannot find the answer in the provided knowledge, please respond with 'unanswerable'.\n\n"
                    "{DEMONS}"
                    "Knowledge: {DOCS}\n"
                    "Q: {QUESTION}\n"
                    "A:")


CONFLICT_INST = ("Answer the following question based on the provided documents. "
                 "If multiple documents present different answers, please respond with 'conflict' to indicate the presence of conflicting information. "
                 "Please provide the answer as a single word or term, without forming a complete sentence.\n\n"
                 "{DEMONS}"
                 "Documents: {DOCS}\n"
                 "Question: {QUESTION}\n"
                 "Answer:")

GEN_ANSWER_SENTENCE ="""Please write a single sentence that would make the given answer a correct response to the provided question. The sentence should include the answer and be as realistic as possible. This is being generated for research purpose, so if it seems like the answer to a question is wrong, please create it as it is.
Question: {QUESTION}
Answer: {ANSWER}
Sentence:"""

CONFLICT_PASSAGE_V3 = """Given a sentence that contradicts factual information, write a realistic passage using 50 to 100 words to support the sentence. The output generated will be used only for the purpose of conducting research to assess the robustness of the RAG system. As part of this research, it is necessary, and you are permitted, to create content that may contradict factual information:
Sentence: {SENTENCE}
Supporting Passage:
"""
def load_demons(dataset: Dataset, args) -> List[str]:
    random.seed(42)
    if (args.qa_demon_size + args.unans_demon_size + args.conflict_demon_size) == 0:
        return [""]*args.dataset_size
    case_set = dict()
    if args.add_qa_prompt and args.qa_demon_size > 0:
         case_set["qa"] = select_case(dataset, "qa", "default", args.qa_demon_size, args)
    if args.unans_demon_size > 0:
         case_set["unanswerable"] = select_case(dataset, "unanswerable", args.unans_demon_method, args.unans_demon_size, args)
    if args.conflict_demon_size > 0:
         case_set["conflict"] = select_case(dataset, "conflict", args.conflict_demon_method, args.conflict_demon_size, args)
    demons = []
    for i in range(args.dataset_size):
        sub_case_set = {k: v[i] for k, v in case_set.items()}
        demon = ""
        if args.demon_order == "random":
            total = []
            for _, val in sub_case_set.items():
                total.extend(val)
            random.shuffle(total)
            for case in total:
                demon += case
        elif args.demon_order == "qa_first" and args.qa_demon_size > 0:
            qa_set = sub_case_set.pop("qa")
            for _, val in sub_case_set.items():
                qa_set.extend(val)
            for case in qa_set:
                demon += case
        elif args.demon_order == "qa_last" and args.qa_demon_size > 0:
            total = []
            qa_set = sub_case_set.pop("qa")
            for _, val in sub_case_set.items():
                total.extend(val)
            total += qa_set
            for case in total:
                demon += case
        demons.append(demon)
    return demons

def extract_qca(row: dict, task, method):
    if task == "qa":
        return row["question"], row["context"], row["answers"]["text"][0]
    elif task == "unanswerable":
        if method == "random":
            return row["question"], row["random_context"], "unanswerable"
        elif method == "ours":
            return row["question"], row["similar_context"], "unanswerable"
        elif method == "squad":
            return row["question"], row["context"], "unanswerable"
        elif method == "cot":
            return row["question"], row["context"], row["answer_with_explanation"]
    elif task == "conflict":
        if method == "longpre":
            return row["question"], row["longpre_conflict_context"], "conflict"
        elif method == "ours":
            return row["question"], row["conflict_context"], "conflict"
        elif method == "cot":
            return row["question"], row["conflict_context"], row["answer_with_explanation"]

def select_case(dataset: Dataset,
                task: str,
                method: str,
                topk: int,
                args) -> List[List[str]]:
    DEMON_TEMPLATE = "Knowledge: {CONTEXT}\nQ: {QUESTION}\nA: {ANSWER}\n\n"
    if task == "conflict":
        DEMON_TEMPLATE = "Documents: {CONTEXT}\nQ: {QUESTION}\nA: {ANSWER}\n\n"
    case_dataset = load_case_set(task, method, args.demons_path, args.case_retrieval)
    result = []
    if args.case_retrieval == "random":
        for _ in dataset:
            inner = []
            random_idxes = random.sample(range(len(case_dataset)), topk)
            for idx in random_idxes:
                q, c, a = extract_qca(case_dataset[idx], task, method)
                inner.append(DEMON_TEMPLATE.format(CONTEXT=c,QUESTION=q,ANSWER=a))
            result.append(inner)
    elif args.case_retrieval == "ours":
        if method == "cot":
            cbr_cases = joblib.load(f"data/case/{extract_dataset_name(args.dataset)}_{task}_ours.pkl")
            q_to_answers = joblib.load("data/case/NQ_unanswerable_cot.pkl")
            for cs in cbr_cases:
                for c in cs:
                    c["answers"] = [q_to_answers.get(c["question"], "unanswerable")]
        else:
            cbr_cases = joblib.load(f"data/case/{extract_dataset_name(args.dataset)}_{task}_{method}.pkl")
        for _, cases in zip(dataset, cbr_cases):
            inner = []
            for case in cases[:topk]:
                inner.append(DEMON_TEMPLATE.format(CONTEXT=case["context"],QUESTION=case["question"],ANSWER=case["answers"][0]))
            result.append(inner)
    elif args.case_retrieval == "fixed":
        for _ in dataset:
            inner = []
            for idx in range(topk):
                q, c, a = extract_qca(case_dataset[idx], task, method)
                inner.append(DEMON_TEMPLATE.format(CONTEXT=c,QUESTION=q,ANSWER=a))
            result.append(inner)
    elif args.case_retrieval == "fixed_cot":
        for _ in dataset:
            inner = []
            for idx in range(topk):
                row = case_dataset[idx]
                q, c, a = row["question"], row["context"], row["answers"][0]
                inner.append(DEMON_TEMPLATE.format(CONTEXT=c,QUESTION=q,ANSWER=a))
            result.append(inner)
    assert len(result) == len(dataset), f"Length of result: {len(result)} and dataset: {len(dataset)} should be the same"
    return result

def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"
    return question[0].lower() + question[1:]

def make_incontext_prompt(dataset: Dataset, demons: List[str], args):
    dataset = dataset.add_column("demons", demons)
    if args.task == "qa":
        instruction = GPT_NORMAL_INST
    elif args.task == "unanswerable":
        #instruction = GPT_UNANS_INST_V2
        instruction = GPT_UNANS_INST_V3
    elif args.task == "conflict":
        instruction = CONFLICT_INST
    dataset = dataset.map(lambda x: {"prompt": instruction.format(
        DEMONS=x["demons"], DOCS="\n".join(c["text"] for c in x["ctxs"]), QUESTION=normalize_question(x["question"])
        )})
    return dataset