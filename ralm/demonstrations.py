from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import torch, spacy
from preprocess import (preprocess_text,
                        query_masking,
                        remove_duplicate,
                        split_sentence_and_make_short_context,
                        query_embedding,
                        remove_duplicate_by_similarity)

def preprocess_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    nlp = spacy.load("en_core_web_trf")
    case_dataset = query_masking(nlp, case_dataset)
    case_dataset = remove_duplicate(case_dataset)
    case_dataset = preprocess_text(case_dataset)
    # case_dataset = split_sentence_and_make_short_context(case_dataset, nlp, args)
    case_dataset = query_embedding(case_dataset)
    case_dataset = remove_duplicate_by_similarity(case_dataset)
    return case_dataset

def main(args: Namespace):
    # rajpurkar/squad
    dataset = load_dataset(args.case_dataset)["train"]
    selected_dataset= dataset.shuffle(seed=42).select(range(100))
    processed_case= preprocess_case(selected_dataset, args)
    processed_case.to_csv(args.output_dir, index= None)
  
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # parser.add_argument("--source_dataset", type=str, required=True, default=None, choices=["nq", "tqa", "webq", "popqa"])
    parser.add_argument("--case_dataset", type=str, required=True, default=None)
    parser.add_argument("--output_dir", type=str, required=False, default="../data/case")
    
    args = parser.parse_args()
    configs = vars(args)
    main(args)
    
    