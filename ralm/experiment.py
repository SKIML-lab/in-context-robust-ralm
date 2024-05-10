import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from llm import gpt_chat_completion
from utils import str2bool
from prompts import load_demon_test, make_incontext_prompt
from evaluation import cal_unans
from preprocess import determine_answerability

def _load_dataset(args):
    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.map(lambda x: {"ctxs" : x["ctxs"][:args.num_ctxs]})
    dataset = dataset.map(lambda x: determine_answerability(x))
    if args.filter_uncertain:
        print("Before filtering :", len(dataset))
        dataset = dataset.filter(lambda x: x["answerable"] != "uncertain")
        print("After filtering uncertain: ", len(dataset))
    if args.test:
        dataset = dataset.shuffle(seed=42).select(range(args.test_size))
    return dataset

def main(dataset: Dataset, args):
    demons = load_demon_test(args)
    dataset = make_incontext_prompt(dataset, demons, args)
    dataset = dataset.map(lambda x: {"pred" : gpt_chat_completion(x["prompt"])}, num_proc=args.gpt_batch_size)
    #dataset = dataset.add_column("pred", ["unanswerable"]*args.dataset_size)
    if args.task == "unans":
        cal_unans(dataset, args)
    elif args.task == "adv_unans":
        pass
    elif args.task == "conflict":
        pass
    else:
        raise ValueError("Invalid task")
    #dataset.push_to_hub(args.output_dir)    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="Atipico1/incontext_nq")
    parser.add_argument("--task", type=str, required=True, default="normal", choices=["noraml", "unans", "conflict"])

    # Configs for demonstrations
    parser.add_argument("--demons_path", type=str, required=False, default="Atipico1/mrqa_v2_unans_1k")
    parser.add_argument("--add_qa_prompt", type=str2bool, required=False, default=True)
    parser.add_argument("--demon_order", type=str, required=False, default="random", choices=["random", "sequential"])
    parser.add_argument("--demons", type=str, required=False, default=None, choices=["random", "ours", "squad", "zeroshot"])
    
    # Size of demonstrations
    parser.add_argument("--qa_demon_size", type=int, required=False, default=0)
    parser.add_argument("--unans_demon_size", type=int, required=False, default=0)
    parser.add_argument("--conflict_demon_size", type=int, required=False, default=0)
    
    # Configs for experiment
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--test_size", type=int, required=False, default=1000)
    parser.add_argument("--gpt_batch_size", type=int, required=False, default=4)
    parser.add_argument("--filter_uncertain", type=str2bool, required=False, default=False)

    # Configs for RAG
    parser.add_argument("--num_ctxs", type=int, required=False, default=5)

    # Configs for 
    parser.add_argument("--output_dir", type=str, required=False, default="../data/case")
    
    args = parser.parse_args()
    configs = vars(args)
    if args.gpt_batch_size > os.cpu_count():
        raise ValueError("Larger batch size than available cores. Please reduce the batch size.")
    dataset = _load_dataset(args)
    args.dataset_size = len(dataset)
    main(dataset, args)
    
    