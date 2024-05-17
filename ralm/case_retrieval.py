import argparse, spacy, joblib
from typing import Dict, List, Tuple
from utils import str2bool
from datasets import load_dataset, Dataset, load_from_disk
from preprocess import query_masking, normalize_answer
from collections import defaultdict
import numpy as np
import joblib
from tqdm.auto import tqdm
from dataset import build_multiple_indexes
from utils import has_answer, SimpleTokenizer, extract_dataset_name

WH_WORDS = ["what", "when", "where", "who", "why", "how","which","whom"]

def match_case(qa_dataset: Dataset, multiple_index, topk: int):
    tokenizer = SimpleTokenizer()
    cnt = 0
    output = []
    for row in tqdm(qa_dataset, desc="CASE Matching..."):
        head_word = row["question"].strip().lower().split()[0]
        if (head_word not in WH_WORDS) or (head_word not in multiple_index.keys()):
            head_word = "original"
        index, id2q = multiple_index[head_word]["index"], multiple_index[head_word]["id2q"]
        query = np.array([row["query_embedding"]]).astype("float32")
        distances, indices = index.search(query, 100)
        cases = []
        for dist, idx in zip(distances[0], indices[0]):
            if len(cases) == topk:
                break
            matched_row = id2q[idx]
            if not _filter_case(matched_row, row["question"], row["answers"], tokenizer):
                continue
            matched_row.update({"distance":str(dist)})
            cases.append(matched_row)
            cnt += 1
            if cnt % (len(qa_dataset) // 5) == 0:
                print("Original Question: ", row["question"])
                for k, v in matched_row.items():
                    print(f"Matched {k}: {v}")
                print("-"*100)
        assert len(cases) == topk, f"Number of cases is not {topk} -> {len(cases)}"
        output.append(cases)
    return output


def make_indexs_for_case(case_dataset: Dataset, key: str, args):
    output = defaultdict(list)
    for row in case_dataset:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        if key == "qa":
            output[head_word].append(({"question":row["question"],
                                        "context":row["context"],
                                        "answers":row["answers"]["text"],
                                        "original_answers":row["answers"]["text"]}, row["query_embedding"]))
        elif key == "unanswerable":
            if args.method == "random":
                ctx_col_name = "random_context"
            elif args.method == "ours":
                ctx_col_name = "similar_context"
            elif args.method == "squad":
                ctx_col_name = "context"
            elif args.method == "cot":
                ctx_col_name = "context"
            output[head_word].append(({"question":row["question"],
                                        "context":row[ctx_col_name],
                                        "answers": ["unanswerable"] if args.method != "cot" else row["answer_with_explanation"],
                                        "original_answers":row["answers"]["text"]}, row["query_embedding"]))
        elif key == "conflict":
            if args.method == "longpre":
                ctx_col_name = "longpre_conflict_context"
            elif args.method == "ours":
                ctx_col_name = "conflict_context"
            elif args.method == "cot":
                ctx_col_name = "conflict_context"
            output[head_word].append(({"question":row["question"],
                                        "context":row["conflict_context"],
                                        "answers": ["conflict"] if args.method != "cot" else row["answer_with_explanation"],
                                        "original_answers":row["answers"]["text"]},row["query_embedding"]))
    return build_multiple_indexes(output, [k for k in output.keys()])

def embedding(texts: List[str]):
    import torch
    from transformers import AutoTokenizer, DPRQuestionEncoder
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda").eval()
    result = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 1024), desc="DPR Embedding..."):
            batch = texts[i:i+1024]
            output = tokenizer(batch,
                            padding="max_length",
                            truncation=True,
                            max_length=64,
                            return_tensors="pt").to("cuda")
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [1024, hidden_dim]
            result.extend([emb for emb in embeddings])
    normalized_arrays = [arr / np.linalg.norm(arr) for arr in result]
    assert len(normalized_arrays) == len(result), "Length doesn't match"
    return normalized_arrays

def _filter_case(case: Dict, question: str, answers: List[str], tokenizer) -> bool:
    if len(case["context"].split()) < 80:
        return False
    if normalize_answer(case["question"]) == normalize_answer(question):
        return False
    for answer in case["original_answers"]:
        for ans in answers:
            if normalize_answer(ans)==normalize_answer(answer):
                return False
    if has_answer(answers, case["context"], tokenizer):
        return False
    return True

def load_case_set(task: str, method: str, case_dataset: str, case_retrieval: str=None) -> Dataset:
    if task == "qa":
        if case_retrieval == "fixed_cot":
            return joblib.load(f"data/case/{task}_cot.pkl")
        return load_dataset(case_dataset, "default")["train"]
    elif task == "unanswerable":
        if case_retrieval == "fixed_cot":
            return joblib.load(f"data/case/{task}_cot.pkl")
        if method == "random":
            return load_dataset(case_dataset, "unanswerable")["train"]
        elif method == "ours":
            return load_dataset(case_dataset, "unanswerable")["train"]
        elif method == "squad":
            return load_dataset(case_dataset, "squad")["train"]
        elif method == "cot":
            ### TEMP CODE ###
            return load_dataset(case_dataset, "unanswerable")["train"]
    elif task == "conflict":
        if case_retrieval == "fixed_cot":
            return joblib.load(f"data/case/{task}_cot.pkl")
        if method == "random":
            return load_dataset(case_dataset, "conflict_word_level")["train"]
        elif method == "ours":
            dataset: Dataset = load_dataset(case_dataset, "conflict")["train"]
            return dataset
        elif method == "cot":
            return load_dataset(case_dataset, "conflict_cot")["train"]
    else:
        raise ValueError("Invalid task")

def main(args):
    case_dataset = load_case_set(args.task, args.method, args.case_dataset)
    try:
        qa_dataset = load_from_disk(f"{args.qa_dataset}_masked")
    except:
        spacy.prefer_gpu(0)
        nlp = spacy.load("en_core_web_trf")
        qa_dataset: Dataset = load_dataset(args.qa_dataset)["train"]
        qa_dataset = query_masking(nlp, qa_dataset)
        qa_dataset = qa_dataset.add_column("query_embedding", embedding(qa_dataset["masked_query"]))
        qa_dataset.save_to_disk(f"{args.qa_dataset}_masked")
    if args.test:
        qa_dataset = qa_dataset.select(range(100))
    nlp = spacy.load("en_core_web_trf")
    case_index = make_indexs_for_case(case_dataset, args.task, args)
    matched_cases = match_case(qa_dataset, case_index, args.num_case)
    joblib.dump(matched_cases, f"{args.save_dir}{extract_dataset_name(args.qa_dataset)}_{args.task}_{args.method}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dataset", type=str, required=True, default="")
    parser.add_argument("--case_dataset", type=str, required=True, default="")
    parser.add_argument("--task", type=str, required=True, default="qa", choices=["qa", "unanswerable", "conflict"])
    parser.add_argument("--num_case", type=int, required=False, default=5)
    parser.add_argument("--case_label", type=str, required=False, default="random")
    parser.add_argument("--method", type=str, required=False, default="default")
    parser.add_argument("--save_dir", type=str, required=False, default="data/case/")
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--printing", type=str2bool, required=False, default=False)
    args = parser.parse_args()
    main(args)