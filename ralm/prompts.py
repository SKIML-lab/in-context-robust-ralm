from datasets import Dataset, load_dataset
from typing import List
import random
from random import randint
GPT_NORMAL_INST = ("Documents:\n{DOCS}\n"
                "Use the above documents to answer the subsequent question.\n"
                "Question: {QUESTION}\n"
                "Answer:")
GPT_UNANS_INST = ("Use the provided documents to answer questions. Please provide the answer as a single word or term, "
                  "without forming a complete sentence. If the answer cannot be found in the documents, write 'I could not find an answer.'\n\n"
                  "{DEMONS}"
                  "Documents:\n{DOCS}\n"
                  "Question: {QUESTION}\n"
                  "Answer:")
GPT_UNANS_INST_V2 = ("Answer the following question based on the provided knowledge. "
                     "If you cannot find the answer in the provided knowledge, please respond with 'unanswerable'. "
                     "Please provide the answer as a single word or term, "
                     "without forming a complete sentence.\n\n"
                    "{DEMONS}"
                    "knowledge:\n{DOCS}\n"
                    "Question: {QUESTION}\n"
                    "Answer:")

LLAMA_NORMAL_INST = ""
LLAMA_UNANS_INST = ""

def load_demon_test(args) -> List[str]:
    case_set = load_dataset(args.demons_path)["train"]
    random.seed(42)
    if args.demons == "zeroshot":
        return [""]*args.dataset_size
    if args.add_qa_prompt:
        qa_output = []
        q, c, a = case_set["question"], case_set["context"], case_set["answer_in_context"]
        for _ in range(args.dataset_size):
            demonstration = ""
            for _ in range(args.qa_demon_size):
                rand_idx = randint(0, len(q)-1)
                demonstration += f"Knowledge:\n{c[rand_idx]}\nQuestion: {q[rand_idx]}\nAnswer: {a[rand_idx][0]}\n\n"
            qa_output.append(demonstration)
    if args.task == "unans":
        unans_output = []
        if args.demons == "random":
            q, c, a = case_set["question"], case_set["random_context"], ['unanswerable']*len(case_set)
        elif args.demons == "ours":
            q, c, a = case_set["question"], case_set["QC_similar_context"], ['unanswerable']*len(case_set)
        elif args.demons == "squad":
            q, c, a = [c["question"] for c in case_set["squad"]], [c["context"] for c in case_set["squad"]], ['unanswerable']*len(case_set)
        else:
            raise NotImplementedError
        for _ in range(args.dataset_size):
            demonstration = ""
            for _ in range(args.unans_demon_size):
                rand_idx = randint(0, len(q)-1)
                demonstration += f"Knowledge:\n{c[rand_idx]}\nQuestion: {q[rand_idx]}\nAnswer: {a[rand_idx]}\n\n"
            unans_output.append(demonstration)
        if args.add_qa_prompt:
            total_output = []
            for qa, unans in zip(qa_output, unans_output):
                if args.demon_order == "random":
                    if randint(0, 1) == 0:
                        total_output.append(unans + qa)
                    else:
                        total_output.append(qa + unans)
                elif args.demon_order == "sequential":
                    total_output.append(qa + unans)
            return total_output
        else:
            return unans_output
    elif args.task == "conflict":
        pass

def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"
    return question[0].lower() + question[1:]

def make_incontext_prompt(dataset: Dataset, demons: List[str], args):
    dataset = dataset.add_column("demons", demons)
    dataset = dataset.map(lambda x: {"prompt": GPT_UNANS_INST_V2.format(
        DEMONS=x["demons"], DOCS="\n".join(c["text"] for c in x["ctxs"]), QUESTION=normalize_question(x["question"])
        )})
    return dataset