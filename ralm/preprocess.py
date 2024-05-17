from datasets import Dataset
from tqdm.auto import tqdm
import re, torch, os, string
from utils import SimpleTokenizer, has_answer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from transformers import AutoTokenizer, DPRQuestionEncoder
# import cupy as cp
import numpy as np
import spacy
NUM_PROC = os.cpu_count()

def _preprocess_context(text):
    # if subset_name == "NewsQA":
    #     start = text.find("--")
    #     end = start + 2
    #     text = text[end:]
    if ("<Table>" in text) or ("<Ol>" in text) or ("<Li>" in text) or ("<Tr>" in text):
        return None
    text = text.replace("<P>", "")
    text = text.replace("</P>", "")
    text = text.replace("[PAR]", "")
    text = text.replace("[DOC]", "")
    text = text.replace("[TLE]", "")
    text = text.replace("[SEP]", "")
    text = re.sub('\n+', '', text)
    text = re.sub(' +', ' ', text)
    return text

def preprocess_text(dataset: Dataset, args) -> Dataset:
    if "mrqa" in args.case_dataset.lower():
        dataset = dataset.map(lambda x: {"context": _preprocess_context(x["context"])}, desc="Preprocessing...")
        print("Before context preprocess: ", len(dataset))
    dataset = dataset.filter(lambda x: len(x["context"].split()) <= 150)
    print("After context preprocess: ", len(dataset))
    return dataset

def query_masking(nlp, dataset: Dataset):
	def masking(doc) -> str:
		if not len(doc.ents):
			return doc.text
		text_list =[]
		for d in doc:
			if d.pos_ == "PUNCT":
				text_list.append("@"+d.text)
			elif d.pos_ == "AUX" and d.text == "'s":
				text_list.append("@"+d.text)
			else:
				text_list.append(d.text)
		for ent in doc.ents:
			text_list[ent.start:ent.end] = ["[B]"]* (ent.end - ent.start)
			text_list[ent.start] = "[MASK]"
		return " ".join(text_list).replace(" [B]", "").replace(" @", "")
	ctxs = dataset["question"]
	result = []
	for i in tqdm(range(0, len(ctxs), 2000), desc="Masking..."):
		batch = ctxs[i:i+2000]
		batch_docs = list(nlp.pipe(batch, batch_size=2000))
		masked_quries = [masking(doc) for doc in batch_docs]
		result.extend(masked_quries)
	assert len(result) == len(ctxs), "Length doesn't match"
	return dataset.add_column("masked_query", result)

def remove_duplicate(data: Dataset):
    masked_queries = data["masked_query"]
    unique_queries = set()
    result_idxs = []
    for idx, query in enumerate(masked_queries):
        if query not in unique_queries:
            unique_queries.add(query)
            result_idxs.append(idx)
    print(f"Remove duplicates by string match -> Before : {len(data)} | After : {len(result_idxs)}")
    filtered_data = data.select(result_idxs, writer_batch_size=50000)
    return filtered_data

def split_sentence_and_make_short_context(dataset: Dataset, nlp):
    simple_tokenizer = SimpleTokenizer()
    answer_passages = []
    answer_sents = []
    answers_in_context = []
    for i in tqdm(range(0, len(dataset), 1024), desc="Splitting sentence..."):
        batch = dataset[i:i+1024]
        docs = list(nlp.pipe(batch["context"], batch_size=1024, disable=["ner"]))
        answers = batch["answers"]
        for doc, answer in zip(docs, answers):
            answer_sent_idx = -1
            sents = [sent.text for sent in doc.sents]
            for idx, sent in enumerate(sents):
                if has_answer(answer, sent, simple_tokenizer):
                    start_idx, end_idx = max(0, idx-3), min(len(sents), idx+4)
                    answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                    while len(answer_passage.split()) < 70:
                        if start_idx == 0 and end_idx == len(sents):
                            break
                        elif start_idx == 0 and end_idx < len(sents):
                            end_idx += 1
                            answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                        elif start_idx > 0 and end_idx == len(sents):
                            start_idx -= 1
                            answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                        else:
                            start_idx -= 1
                            end_idx += 1
                            answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                    answer_sents.append(sent)
                    buffer = []
                    for ans in answer:
                        if has_answer([ans], answer_passage, simple_tokenizer):
                            buffer.append(ans)
                    if buffer == []:
                        print(f"Answer not found in context!\nAnswer : {answer}\nContext : {answer_passage}")
                        answer_passages.append(None)
                        answers_in_context.append(None)
                    else:
                        answers_in_context.append(buffer)
                        answer_passages.append(answer_passage)
                    answer_sent_idx = idx
                    break
            if answer_sent_idx == -1:
                answer_sents.append(None)
                answers_in_context.append(None)
                answer_passages.append(None)
                
    assert len(list(set([len(answer_passages), len(dataset), len(answer_sents), len(answers_in_context)])))==1, "Length doesn't match"
    print("Before split: ", len(dataset))
    dataset = dataset.add_column("short_context", answer_passages)
    dataset = dataset.add_column("answer_sent", answer_sents)
    dataset = dataset.add_column("answer_in_context", answers_in_context)
    dataset = dataset.filter(lambda x: x["short_context"] is not None, num_proc=NUM_PROC)
    print("After split: ", len(dataset))
    dataset = dataset.filter(lambda x: len(x["short_context"].split())< 120 and len(x["short_context"].split()) > 70, num_proc=NUM_PROC)
    print("After context length filtering: ", len(dataset))
    dataset = dataset.filter(lambda x: all([len(ans.split())<= 7 for ans in x["answer_in_context"]]), num_proc=NUM_PROC, desc="Max answer len filtering...")
    print("After answer length filtering: ", len(dataset))
    return dataset

def query_embedding(dataset: Dataset):
    queries = dataset["masked_query"]
    result = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
    model.eval()

    for i in tqdm(range(0, len(queries), 1024), desc="Embedding..."):
        batch = queries[i:i+1024]
        output = tokenizer(batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [1024, hidden_dim]
        result.extend([emb for emb in embeddings])
    
    # Normalize the embeddings
    normalized_arrays = [emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb for emb in result]
    assert len(normalized_arrays) == len(queries), "Length doesn't match"
    return dataset.add_column("query_embedding", normalized_arrays)


def remove_duplicate_by_similarity(dataset: Dataset):
    questions = dataset["question"]
    result = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device="cuda")
    model.max_seq_length = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), 1024), desc="Embedding..."):
            batch = questions[i:i+1024]
            result.extend(model.encode(batch, batch_size=1024, show_progress_bar=False))
    torch.cuda.empty_cache()
    matrix = np.array([v/np.linalg.norm(v) for v in result], dtype=np.float16)
    key_matrix_indices = []
    for i in tqdm(range(len(matrix)), desc="Removing duplicates by similarity..."):
        if len(key_matrix_indices) == 0:
            key_matrix_indices.append(i)
        else:
            current_matrix = matrix[i:i+1]
            key_matrix_subset = matrix[key_matrix_indices]
            similarity = np.dot(current_matrix, key_matrix_subset.T)
            if not np.any(similarity >= 0.9):
                key_matrix_indices.append(i)
    print(f"Remove duplicates by similarity-> Before : {len(dataset)} | After : {len(key_matrix_indices)}")
    dataset = dataset.select(key_matrix_indices, writer_batch_size=50000)
    return dataset

def determine_answerability(ex) -> dict:
    def hasanswer(ctxs) -> bool:
        return any([c["hasanswer"] for c in ctxs])
    def answerable(ctxs) -> bool:
        res = []
        for ctx in ctxs:
            hasanswer, entail = ctx["hasanswer"], ctx["nli"]
            ### Q - A -> Answer sentence
            ### Answer sentence -> hypothesis
            ### Retrieved context -> premise
            ### NLI (premise, hypothesis) -> entailment
            if hasanswer and (entail in ["entailment", "contradiction"]):
                res.append("answerable")
            elif (not hasanswer) and (entail != "entailment"):
                res.append("unanswerable")
            else:
                res.append("uncertain")
        if res.count("answerable") >= 1:
            return "answerable"
        elif res.count("unanswerable") == 5:
            return "unanswerable"
        else:
            return "uncertain"
    return {"answerable": answerable(ex["ctxs"]), "hasanswer": hasanswer(ex["ctxs"])}

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