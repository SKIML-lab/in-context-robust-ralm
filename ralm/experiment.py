import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from llm import gpt_chat_completion, vllm_completion, gpt_completion
from utils import str2bool, validate_args
from prompts import load_demons, make_incontext_prompt
from evaluation import cal_unans, cal_conflict
from preprocess import determine_answerability
from dataset import perturbate_context

def _load_dataset(args):
    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.map(lambda x: {"ctxs" : x["ctxs"][:args.num_ctxs]})
    dataset = dataset.map(lambda x: determine_answerability(x))
    dataset = dataset.map(lambda x: perturbate_context(x, args))
    if args.filter_uncertain:
        print("Before filtering :", len(dataset))
        dataset = dataset.filter(lambda x: x["answerable"] != "uncertain")
        print("After filtering uncertain: ", len(dataset))
    if args.test:
        dataset = dataset.shuffle(seed=42).select(range(args.test_size))
    return dataset

def main(dataset: Dataset, args):
    demons = load_demons(dataset, args)
    dataset = make_incontext_prompt(dataset, demons, args)
    if args.dummy_data:
        dataset = dataset.map(lambda x: {"pred" : "unanswerable" if args.task == "unanswerable" else "conflict"}, num_proc=args.gpt_batch_size)
    else:
        if args.llm == "chatgpt":
            dataset = dataset.map(lambda x: {"pred" : gpt_chat_completion(x["prompt"], args)}, num_proc=args.gpt_batch_size)
        elif args.llm == "chatgpt_inst":
            dataset = dataset.map(lambda x: {"pred" : gpt_completion(x["prompt"], args)}, num_proc=args.gpt_batch_size)
        else:
            output = vllm_completion(dataset["prompt"], args)
            dataset = dataset.add_column("pred", output)
    if args.task == "unanswerable":
        df = cal_unans(dataset, args)
    elif args.task == "adv_unanswerable":
        pass
    elif args.task == "conflict":
        df = cal_conflict(dataset, args)
    else:
        raise ValueError("Invalid task")
    #dataset.push_to_hub(args.output_dir)    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="Atipico1/incontext_nq")
    parser.add_argument("--task", type=str, required=True, default="normal", choices=["noraml", "unanswerable", "conflict"])

    # Configs for demonstrations
    parser.add_argument("--demons_path", type=str, required=False, default="Atipico1/mrqa_v2_unans_1k")
    parser.add_argument("--add_qa_prompt", type=str2bool, required=False, default=True)
    parser.add_argument("--demon_order", type=str, required=False, default="random", choices=["qa_first", "qa_last", "random"])
    parser.add_argument("--case_retrieval", type=str, required=False, default="random", choices=["random", "ours", "fixed", "fixed_cot"])
    
    # Methods
    parser.add_argument("--unans_demon_method", type=str, required=False, default="", choices=["random", "ours", "squad", "cot"])
    parser.add_argument("--conflict_demon_method", type=str, required=False, default="", choices=["random", "ours", "cot"])

    # Size of demonstrations
    parser.add_argument("--qa_demon_size", type=int, required=False, default=0)
    parser.add_argument("--unans_demon_size", type=int, required=False, default=0)
    parser.add_argument("--conflict_demon_size", type=int, required=False, default=0)
    
    # Configs for experiment
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--dummy_data", type=str2bool, required=False, default=False)
    parser.add_argument("--test_size", type=int, required=False, default=1000)
    parser.add_argument("--gpt_batch_size", type=int, required=False, default=4)
    parser.add_argument("--filter_uncertain", type=str2bool, required=False, default=False)

    # Configs for RAG
    parser.add_argument("--llm", type=str, required=False, default="chatgpt")
    parser.add_argument("--num_ctxs", type=int, required=False, default=5)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=100)

    # Configs for save
    parser.add_argument("--output_dir", type=str, required=False, default="./data/case")
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--project_prefix", type=str, required=False, default="")
    
    args = parser.parse_args()
    configs = vars(args)
    #validate_args(args)
    if args.gpt_batch_size > os.cpu_count():
        raise ValueError("Larger batch size than available cores. Please reduce the batch size.")
    dataset = _load_dataset(args)
    args.dataset_size = len(dataset)
    main(dataset, args)
    
    