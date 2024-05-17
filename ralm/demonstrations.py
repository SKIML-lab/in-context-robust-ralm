from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from utils import str2bool, has_answer, SimpleTokenizer, filter_and_map
from datasets import load_dataset, Dataset, concatenate_datasets
from dataset import (find_unanswerable_contexts,
                    find_random_contexts,
                    make_entity_set,
                    detect_entity_type,
                    gen_entity_vector,
                    build_index_with_ids,
                    find_random_entity,
                    find_similar_entity,
                    cal_cosine_similarities,
                    update_context_with_substitution_string)
from tqdm.auto import tqdm
import torch, spacy, joblib
import numpy as np
from vllm import LLM, SamplingParams
from preprocess import (preprocess_text,
                        query_masking,
                        remove_duplicate,
                        split_sentence_and_make_short_context,
                        query_embedding,
                        remove_duplicate_by_similarity)
from prompts import GEN_ANSWER_SENTENCE, CONFLICT_PASSAGE_V3
from llm import LLAMA3_CHAT


def generate_unanswerable_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    if "squad_v2" in args.case_dataset.lower():
        case_dataset.push_to_hub(args.output_dir, config_name="squad")
        return case_dataset
    case_dataset = find_unanswerable_contexts(case_dataset, args)
    case_dataset = find_random_contexts(case_dataset)
    case_dataset.push_to_hub(args.output_dir, config_name="unanswerable")
    return case_dataset

def generate_answer_sentence(case_dataset: Dataset, args: Namespace) -> Dataset:
    llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct",
              gpu_memory_utilization= 0.95,
              max_context_len_to_capture=1024,
              seed=42,
              max_model_len=1024,
              swap_space=4,
              tensor_parallel_size=torch.cuda.device_count())
    config = SamplingParams(top_p=0.9, max_tokens=128)
    case_dataset = case_dataset.map(lambda x: {"prompt": GEN_ANSWER_SENTENCE.format(QUESTION=x["question"], ANSWER=x["answers"]["text"][0])})
    case_dataset = case_dataset.map(lambda x: {"input": LLAMA3_CHAT.format(PROMPT=x["prompt"])})
    output = llm.generate(case_dataset["input"], sampling_params=config)
    output = [o.outputs[0].text.strip() for o in output]
    output = [o.split("<|eot_id|>")[0] for o in output]
    case_dataset = case_dataset.add_column("answer_sentence", output)
    case_dataset.push_to_hub(args.case_dataset, config_name="answer_sentence")
    return case_dataset

def generate_conflict_passage(case_dataset: Dataset, args:  Namespace) -> Dataset:
    llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct",
              gpu_memory_utilization= 0.95,
              max_context_len_to_capture=1024,
              seed=42,
              max_model_len=1024,
              swap_space=4,
              tensor_parallel_size=torch.cuda.device_count())
    config = SamplingParams(top_p=0.9, max_tokens=256)
    tokenizer = SimpleTokenizer()
    case_dataset = case_dataset.filter(lambda x: (x["similar_entity"] is not None) and (has_answer(x["answers"], x["answer_sentence"], tokenizer)))
    case_dataset = case_dataset.map(lambda x: {"conflict_sentence": update_context_with_substitution_string(x["answer_sentence"], x["answers"], x["similar_entity"])})
    case_dataset = case_dataset.filter(lambda x: has_answer(x["similar_entity"], x["conflict_sentence"], tokenizer))
    case_dataset = case_dataset.filter(lambda x: not (has_answer(x["answers"], x["conflict_sentence"], tokenizer)))
    case_dataset = case_dataset.map(lambda x: {"prompt" : CONFLICT_PASSAGE_V3.format(SENTENCE=x["conflict_sentence"])})
    case_dataset = case_dataset.map(lambda x: {"input": LLAMA3_CHAT.format(PROMPT=x["prompt"])})
    output = llm.generate(case_dataset["input"], sampling_params=config)
    output = [o.outputs[0].text.strip() for o in output]
    output = [o.split("<|eot_id|>")[0] for o in output]
    case_dataset = case_dataset.add_column("conflict_context", output)
    if args.test:
        case_dataset.push_to_hub("Atipico1/conflict_passage_test")
    else:
        case_dataset.push_to_hub(args.case_dataset, config_name="conflict")
    return case_dataset

def generate_adversarial_passage(case_dataset: Dataset, args: Namespace) -> Dataset:
    tokenizer = SimpleTokenizer()
    case_dataset = case_dataset.filter(lambda x: has_answer(x["answers"], x["answer_sentence"], tokenizer))
    

def generate_conflict_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    if args.test:
        case_dataset = case_dataset.select(range(args.test_size))
    entities_groupby_type = joblib.load("data/entity/entity_group.pkl")
    entity_vector_groupby_type = joblib.load("data/entity/entity_group_vec.pkl")
    case_dataset = detect_entity_type(case_dataset, "answer_sentence")
    case_dataset = case_dataset.filter(lambda x: x["entity_type"])
    case_dataset = gen_entity_vector(case_dataset, "answers")
    index_per_entity = {}
    for k, v in entity_vector_groupby_type.items():
        v = np.array(v).astype('float32')
        index_per_entity[k] = build_index_with_ids(v, save_dir="", name=k, is_save=False, gpu_id=-100) ## GPU 때문에 score 에러
    similar_entities, similar_scores = find_similar_entity(case_dataset, args, entities_groupby_type, index_per_entity)
    case_dataset = case_dataset.add_column("similar_entity",  similar_entities)
    case_dataset = case_dataset.add_column("similar_entity_score", similar_scores)
    random_entities = find_random_entity(case_dataset, entities_groupby_type, args)
    case_dataset = case_dataset.add_column("random_entity", random_entities)
    random_scores = cal_cosine_similarities(case_dataset["entity_vector"], case_dataset["random_entity"], args)
    case_dataset = case_dataset.add_column("random_entity_score", random_scores)
    case_dataset = case_dataset.remove_columns(["entity_vector"])
    if args.test:
        case_dataset.push_to_hub("Atipico1/conflict_test")
    else:
        case_dataset.push_to_hub(f"{args.case_dataset}_ent")
    return case_dataset

def preprocess_case(case_dataset: Dataset, args: Namespace) -> Dataset:
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    case_dataset = preprocess_text(case_dataset, args)
    case_dataset = query_masking(nlp, case_dataset)
    case_dataset = remove_duplicate(case_dataset)
    # case_dataset = split_sentence_and_make_short_context(case_dataset, nlp, args)
    case_dataset = query_embedding(case_dataset)
    #case_dataset = remove_duplicate_by_similarity(case_dataset)
    case_dataset.push_to_hub(args.output_dir)
    return case_dataset

def main(args: Namespace):
    dataset = load_dataset(args.case_dataset)
    if len(dataset.keys()) > 1:
        new_dataset = []
        for key in dataset.keys():
            new_dataset.append(dataset[key])
        dataset = concatenate_datasets(new_dataset)
    else:
        dataset = dataset["train"]
    if args.task == "preprocess":
        if "squad_v2" in args.case_dataset.lower():
            dataset = dataset.filter(lambda x: len(x["answers"]["text"]) == 0)
        processed_case = preprocess_case(dataset, args)
    elif args.task == "unanswerable":
        unanswerable_case = generate_unanswerable_case(dataset, args)
    elif args.task == "conflict":
        try:
            dataset = load_dataset(args.case_dataset, "answer_sentence")["train"]
            print("Answer sentence already generated")
        except:
            print("Generate answer sentence first")
            dataset = load_dataset(args.case_dataset)["train"]
            dataset = generate_answer_sentence(dataset, args)
        try:
            conflict_case = load_dataset(f"{args.case_dataset}_ent")["train"]
        except:
            conflict_case = generate_conflict_case(dataset, args)
        conflict_passage = generate_conflict_passage(conflict_case, args)
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
    
    