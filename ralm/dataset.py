from datasets import Dataset, load_dataset
import spacy, re
from tqdm.auto import tqdm
from utils import SimpleTokenizer, has_answer
import numpy as np
import faiss, os, joblib, random
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
from preprocess import normalize_answer
from typing import List, Dict
from collections import defaultdict

def dpr_embed(dataset: Dataset, col: str, args) -> list[np.ndarray]: 
    import torch    
    inputs = list(set(dataset[col]))
    if not args.test:
        if os.path.exists(f"data/index/{args.case_dataset.split('/')[-1]}_{col}_embeddings.pkl"):
            cached_array = joblib.load(f"data/index/{args.case_dataset.split('/')[-1]}_{col}_embeddings.pkl")
            if len(cached_array) == len(inputs):
                print(f"{col} embeddings already exist")
                return cached_array, inputs
    result = []
    if col == "question":
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to("cuda")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), 1024), desc="DPR Embedding..."):
            batch = inputs[i:i+1024]
            output = tokenizer(batch,
                            padding="max_length",
                            truncation=True,
                            max_length=64 if col == "question" else 256,
                            return_tensors="pt").to("cuda")
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [1024, hidden_dim]
            result.extend([emb for emb in embeddings])
    normalized_arrays = [arr / np.linalg.norm(arr) for arr in result]
    assert len(normalized_arrays) == len(result), "Length doesn't match"
    if not args.test:
        joblib.dump(normalized_arrays, f"data/index/{args.case_dataset.split('/')[-1]}_{col}_embeddings.pkl")
        print(f"{col} embeddings saved")
    return normalized_arrays, inputs

def build_multiple_indexes(input_dict: Dict, subsets: List[str]):
    output = dict()
    for subset in subsets:
        if (input_dict.get(subset) == []) or (input_dict.get(subset) == None):
            continue
        else:
            rows, embeddings = [d[0] for d in input_dict[subset]], [d[1] for d in input_dict[subset]]
            index = build_index_with_ids(np.array(embeddings), "data/index/", subset, is_save=False, gpu_id=-100)
            output.update({subset:{"index":index, "id2q":{_id:row for _id, row in zip(np.arange(len(embeddings)).astype('int64'), rows)}}})
    return output

def build_index_with_ids(vectors: np.ndarray, save_dir: str, name: str, is_save: bool = True, gpu_id: int =0):
    index_flat = faiss.IndexFlatIP(len(vectors[0]))
    index = faiss.IndexIDMap(index_flat)
    ids = np.arange(len(vectors)).astype('int64')
    if gpu_id != -100 and faiss.get_num_gpus() != 0:
        if faiss.get_num_gpus() > 1:
            gpu_index = faiss.index_cpu_to_all_gpus(index)
        else:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            gpu_index.add_with_ids(vectors, ids)
        return gpu_index
    else:
        index.add_with_ids(vectors, ids)
        return index

def find_unanswerable_contexts(dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    q_embedding, questions = dpr_embed(dataset=dataset, col="question", args=args)
    c_embedding, contexts = dpr_embed(dataset=dataset, col="context", args=args)
    q_embedding, c_embedding = np.array(q_embedding).astype("float32"), np.array(c_embedding).astype('float32')
    index = build_index_with_ids(c_embedding, save_dir="/data/seongilpark/index", name="context", is_save=False)
    new_context = []
    D, I = index.search(q_embedding, args.topk)
    distances, nearest_neighbors = D[:, 1:], I[:, 1:] # [len(contexts), topk-1]
    for dists, neighbors, answer in tqdm(zip(distances, nearest_neighbors, dataset["answers"]), desc="Faiss contexts searching...", total=len(nearest_neighbors)):
        is_valid = False
        for idx, dist in zip(neighbors, dists):
            if (not has_answer(answer, contexts[idx], simple_tokenizer)) and (dist < 0.9):
                new_context.append(contexts[idx])
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers: {answer}")
            new_context.append(None)
    assert len(new_context) == len(dataset), f"Length doesn't match {len(new_context)} != {len(dataset)}"
    dataset = dataset.add_column("similar_context", new_context)
    dataset = dataset.filter(lambda x: x["similar_context"] is not None, num_proc=8)
    return dataset

def find_similar_questions(dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding, questions = dpr_embed(dataset=dataset, col="question", args=args)
    embedding = np.array(embedding).astype("float32")
    index = build_index_with_ids(embedding, "./data/index", "question", is_save=False)
    assert len(questions) == len(dataset), "There is a not unique question in the dataset"
    new_context = []
    answers = dataset["answers"]
    contexts = dataset["context"]

    _, I = index.search(embedding, args.topk)
    nearest_neighbors = I[:, 1:] # [len(query_vectors), topk-1]
    for neighbors, answer in tqdm(zip(nearest_neighbors, answers), desc="Faiss question searching...", total=len(nearest_neighbors)):
        is_valid = False
        for idx in neighbors:
            if not has_answer(answer, contexts[idx], simple_tokenizer):
                new_context.append(contexts[idx])
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers: {answer}")
            new_context.append(None)
    assert len(new_context) == len(dataset), f"Length doesn't match {len(new_context)} != {len(dataset)}"
    dataset = dataset.add_column("Q_similar_context", new_context)
    dataset = dataset.filter(lambda x: x["Q_similar_context"] is not None, num_proc=8)
    return dataset, embedding

def find_similar_contexts(dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding, contexts = dpr_embed(dataset=dataset, col="context", args=args)
    embedding = np.array(embedding).astype('float32')
    c2embs = {c:emb for c, emb in zip(contexts, embedding)}
    c2ids = {c:i for i, c in enumerate(contexts)}
    index = build_index_with_ids(embedding, save_dir="/data/seongilpark/index", name="context", is_save=False)
    old2new_context = dict()
    _, I = index.search(embedding, args.topk)
    nearest_neighbors = I[:, 1:] # [len(contexts), topk-1]
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Faiss contexts searching..."):
        query_context_id = c2ids[row["context"]]
        answers = row["answers"]
        neighbors = nearest_neighbors[query_context_id]
        is_valid = False
        for idx in neighbors:
            if not has_answer(answers, contexts[idx], simple_tokenizer):
                old2new_context[row["context"]] = contexts[idx]
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers : {row['answers']}")
            old2new_context[row["context"]] = None
    assert len(old2new_context) == len(contexts), f"Length doesn't match {len(old2new_context)} != {len(contexts)}"
    dataset = dataset.map(lambda x: {"C_similar_context": old2new_context[x["context"]]}, num_proc=8)
    dataset = dataset.filter(lambda x: x["C_similar_context"] is not None, num_proc=8)
    return dataset, c2embs

def find_random_contexts(dataset: Dataset):
    simple_tokenizer = SimpleTokenizer()
    contexts = list(set(dataset["context"]))
    new_context = []
    for row in tqdm(dataset, desc="Finding random contexts..."):
        max_count = 0
        a = row["answers"]
        random_ctx = random.choice(contexts)
        while has_answer(a, random_ctx, simple_tokenizer) and max_count < 10:
            random_ctx = random.choice(contexts)
            max_count += 1
        if max_count < 10:
            new_context.append(random_ctx)
        else:
            new_context.append(None)
    assert len(new_context) == len(dataset), "Length doesn't match"
    dataset = dataset.add_column("random_context", new_context)
    dataset = dataset.filter(lambda x: x["random_context"] is not None, num_proc=8)
    return dataset

def check_alias(entity: str, answers: List[str]):
    if isinstance(answers, dict):
        answers = answers["text"]
    for answer in answers:
        if normalize_answer(entity) in normalize_answer(answer) or normalize_answer(answer) in normalize_answer(entity):
            return True
    return False

def find_similar_entity(dataset: Dataset, args, entities_groupby_type: Dict[str, List], indexs: Dict):
    output = []
    scores = []
    for row in tqdm(dataset, desc="Find similar entity", total=len(dataset)):
        entity_type = row["entity_type"]
        index = indexs[entity_type]
        entity_set = entities_groupby_type[entity_type]
        query = np.array([row["entity_vector"]]).astype("float32")
        distances, indices = index.search(query, 100)
        hit = False
        for score, idx in zip(distances[0], indices[0]):
            if score < 0.8:
                if check_alias(entity_set[idx], row["answers"]):
                    continue
                hit = True
                output.append(entity_set[idx])
                scores.append(score)
                break
        if not hit:
            if check_alias(entity_set[indices[0][-1]], row["answers"]):
                output.append(None)
                scores.append(None)
            else:
                output.append(entity_set[indices[0][-1]])
                scores.append(distances[0][-1])
    return output, scores       

def find_random_entity(dataset: Dataset, entities_groupby_type: Dict[str, List], args):
    output = []
    for row in tqdm(dataset, desc="Find random entity", total=len(dataset)):
        max_cnt = 0
        entity_type = row["entity_type"]
        entity_set = entities_groupby_type[entity_type]
        random_entity = random.choice(entity_set)
        while check_alias(random_entity, row["answers"]) and max_cnt < 50:
            max_cnt += 1
            random_entity = random.choice(entity_set)
        if max_cnt >= 50:
            output.append(None)
        else:
            output.append(random_entity)
    return output

def is_valid_entity(entity: str, label: str) -> bool:
    if "@" in entity:
        return False
    if label == "DATE" and "through" in entity:
        return False
    if label == "DATE" and "and" in entity:
        return False
    return True

def make_entity_set(args) -> None:
    import torch
    from collections import defaultdict
    spacy.prefer_gpu()
    dataset = load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    dataset = dataset.filter(lambda x: len(x["text"]) > 50, num_proc=os.cpu_count())
    dataset = dataset.map(lambda x: {"text": x["text"].strip()}, num_proc=os.cpu_count())
    contexts= dataset["text"]
    entities = defaultdict(list)
    nlp = spacy.load("en_core_web_trf")
    for i in tqdm(range(0, len(contexts), 2000), desc="Extracting entities from entity pool..."):
        batch = contexts[i:i+2000]
        docs = list(nlp.pipe(batch, batch_size=2000))
        for doc in docs:
            if doc.ents:
                for ent in doc.ents:
                    if is_valid_entity(ent.text.lower(), ent.label_):
                        entities[ent.label_].append(ent.text)
    entities = {k: list(set(v)) for k, v in entities.items()}
    nlp = spacy.load("en_core_web_lg")
    entities_to_vector = defaultdict(list)
    valid_entity_group = defaultdict(list)
    for k, v in entities.items():
        vectors, valid_entities, valid_vectors = [], [], []
        for i in range(0, len(v), 2000):
            batch = v[i:i+2000]
            docs = list(nlp.pipe(batch, batch_size=2000))
            if torch.cuda.is_available():
                vectors.extend([doc.vector.get() for doc in docs])
            else:
                vectors.extend([doc.vector for doc in docs])
        vectors = [v/np.linalg.norm(v) for v in vectors]
        for vec, ent in zip(vectors, v):
            if not np.isnan(vec).any():
                valid_entities.append(ent)
                valid_vectors.append(vec)
        entities_to_vector[k] = np.array(valid_vectors)
        valid_entity_group[k] = valid_entities
    print("Entity pool extracted!")
    print("Entity group size:\n")
    for k, v in valid_entity_group.items():
        print(f"{k}: {len(v)}")
    if not args.test:
        joblib.dump(entities_to_vector, f"./data/entity/entity_group_vec.pkl")
        joblib.dump(valid_entity_group, f"./data/entity/entity_group.pkl")
        print("Entity pool saved!")
        print("Saved path: ./data/entity/entity_group_vec.pkl")
    return valid_entity_group, entities_to_vector

def detect_entity_type(dataset: Dataset, col_name: str):
    spacy.prefer_gpu()
    res = []
    nlp = spacy.load("en_core_web_trf")
    entities = defaultdict(list)
    inputs = dataset[col_name]
    answers = dataset["answers"]
    for i in tqdm(range(0, len(inputs), 2048), desc="Detecting entity types..."):
        batch = inputs[i:i+2048]
        batch_answers = answers[i:i+2048]
        docs = list(nlp.pipe(batch, batch_size=2048))
        for doc, ans in zip(docs, batch_answers):
            hit = False
            if doc.ents:
                for ent in doc.ents:
                    entities[ent.label_].append(ent.text)
                    if not hit and normalize_answer(ent.text) == normalize_answer(ans["text"][0]):
                        res.append(ent.label_)
                        hit = True
            if not hit:
                res.append(None)
    dataset = dataset.add_column("entity_type", res)
    return dataset

def detect_entities(dataset: Dataset, col_name: str):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    validated_inputs = [x if x is not None else "" for x in dataset[in_col]]
    docs = list(nlp.pipe(validated_inputs))
    ents, ents_count = [],[]
    for doc in docs:
        if doc.ents:
            ents.append(", ".join(['"'+ent.text+'"' for ent in doc.ents]))
            ents_count.append(len(doc.ents))
        else:
            ents.append(None)
            ents_count.append(0)
    dataset = dataset.add_column(out_col, ents)
    dataset = dataset.add_column(out_col+"_count", ents_count)
    return dataset

def gen_entity_vector(dataset: Dataset, col_name: str):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")
    entities = dataset[col_name]
    if isinstance(entities[0], list):
        entities = [ent[0] for ent in entities]
    elif isinstance(entities[0], dict):
        entities = [ent["text"][0] for ent in entities]
    vectors = []
    for i in tqdm(range(0, len(entities), 2048), desc="Generating entity vector..."):
        batch = entities[i:i+2048]
        docs = list(nlp.pipe(batch, batch_size=2048))
        vectors.extend([doc.vector.get() for doc in docs])
    vectors = [v/np.linalg.norm(v) if not np.isnan(v).any() else None for v in vectors]
    dataset = dataset.add_column("entity_vector", vectors)
    return dataset

def cal_cosine_similarities(queries_vector: List[np.ndarray], entities: List[str], args):
    def cosine_similarity(a, b):
        if not isinstance(b, np.ndarray):
            b = b.get()
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    import spacy
    scores = []
    nlp = spacy.load("en_core_web_lg")
    assert len(queries_vector) == len(entities), "Length of queries and entities should be same"
    for query_vec, entity in tqdm(
        zip(queries_vector, entities), desc="Calculating cosine similarity", total=len(queries_vector)
        ):
        if not entity:
            scores.append(None)
            continue
        doc = nlp(entity)
        scores.append(cosine_similarity(query_vec, doc.vector))
    return scores

def find_answer_in_context(answer_text: str, context: str):
    if isinstance(context, str):
        context_spans = [
            (m.start(), m.end())
            for m in re.finditer(re.escape(answer_text.lower()), context.lower())
        ]
        return context_spans
    else:
        return [""]

def update_context_with_substitution_string(
    context: str, originals:List[str], substitution: str, replace_every_string=True
) -> str:
    replace_spans = []
    if isinstance(originals, dict):
        originals = originals["text"]
    for orig_answer in originals:
        replace_spans.extend(find_answer_in_context(orig_answer, context))
    replace_strs = set([context[span[0] : span[1]] for span in replace_spans])
    for replace_str in replace_strs:
        context = context.replace(replace_str, substitution)
    return context

def perturbate_context(ex: dict, args):
    if args.task != "conflict":
        return {"ctxs": ex["ctxs"]}
    random.seed(42)
    ctxs = ex["ctxs"]
    if ex["is_valid_conflict_passage"]:
        ctxs.insert(random.randint(0, len(ctxs)), {"text": ex["gpt_conflict_passage"][0], "hasanswer": False, "is_conflict": True})
        return {"ctxs": ctxs}
    else:
        return {"ctxs": ctxs}
