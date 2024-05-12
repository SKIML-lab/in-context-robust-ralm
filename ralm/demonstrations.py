from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from utils import str2bool
from datasets import load_dataset, Dataset, concatenate_datasets
from dataset import find_unanswerable_contexts, find_random_contexts, make_entity_set
from tqdm.auto import tqdm
import torch, spacy
from preprocess import (preprocess_text,
                        query_masking,
                        remove_duplicate,
                        split_sentence_and_make_short_context,
                        query_embedding,
                        remove_duplicate_by_similarity)

def generate_unanswerable_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    case_dataset = find_unanswerable_contexts(case_dataset, args)
    case_dataset = find_random_contexts(case_dataset)
    case_dataset.push_to_hub(args.output_dir, config_name="unanswerable")
    return case_dataset

def generate_conflict_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    pass

def preprocess_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    case_dataset = preprocess_text(case_dataset, args)
    case_dataset = query_masking(nlp, case_dataset)
    case_dataset = remove_duplicate(case_dataset)
    # case_dataset = split_sentence_and_make_short_context(case_dataset, nlp, args)
    case_dataset = query_embedding(case_dataset)
    case_dataset = remove_duplicate_by_similarity(case_dataset)
    case_dataset.push_to_hub(args.output_dir)
    return case_dataset

def main(args: Namespace):
    # rajpurkar/squad
    dataset = load_dataset(args.case_dataset)
    if len(dataset.keys()) > 1:
        new_dataset = []
        for key in dataset.keys():
            new_dataset.append(dataset[key])
        dataset = concatenate_datasets(new_dataset)
    else:
        dataset = dataset["train"]
    if args.task == "preprocess":
        processed_case = preprocess_case(dataset, args)
        #processed_case.to_csv(args.output_dir, index= None)
    elif args.task == "unanswerable":
        unanswerable_case = generate_unanswerable_case(dataset, args)
        #unanswerable_case.to_csv(args.output_dir, index= None)
    elif args.task == "conflict":
        conflict_case = generate_conflict_case(dataset, args)
        #conflict_case.to_csv(args.output_dir, index= None)
    elif args.task == "gen_entity":
        make_entity_set(args)
  
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # parser.add_argument("--source_dataset", type=str, required=True, default=None, choices=["nq", "tqa", "webq", "popqa"])
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--test_size", type=int, required=False, default=1000)
    parser.add_argument("--task", type=str, required=True, default=None, choices=["preprocess", "unanswerable", "conflict", "gen_entity"])
    parser.add_argument("--case_dataset", type=str, required=True, default=None)
    parser.add_argument("--topk", type=int, required=False, default=10)
    parser.add_argument("--output_dir", type=str, required=False, default="../data/case")
    
    args = parser.parse_args()
    configs = vars(args)
    main(args)
    
    