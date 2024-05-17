from datasets import Dataset
import pandas as pd
import wandb
from utils import has_answer, SimpleTokenizer, make_run_name, exact_match_score, f1_score

def cal_unans(dataset: Dataset, args):
    tokenizer = SimpleTokenizer()
    df = dataset.to_pandas()
    df = df.drop(columns=["ctxs"])
    answerable_ratio = df["answerable"].value_counts(normalize=True).get("answerable", 0)
    unanswerable_ratio = df["answerable"].value_counts(normalize=True).get("unanswerable", 0)

    sub_df = df[df["answerable"] != "uncertain"]
    sub_df["answerable_acc"] = sub_df.apply(lambda x: has_answer(x["answers"] if x["answerable"] == "answerable" else ['unanswerable'], x["pred"], tokenizer), axis=1)
    sub_df["answerable_em"] = sub_df.apply(lambda x: exact_match_score(x["pred"], x["answers"] if x["answerable"] == "answerable" else ['unanswerable']), axis=1)
    sub_df["answerable_f1"] = sub_df.apply(lambda x: f1_score(x["pred"], x["answers"] if x["answerable"] == "answerable" else ['unanswerable']), axis=1)
    answerable_acc, unanswerable_acc = sub_df[sub_df["answerable"] == "answerable"]["answerable_acc"].mean(), sub_df[sub_df["answerable"] == "unanswerable"]["answerable_acc"].mean()
    answerable_em, unanswerable_em = sub_df[sub_df["answerable"] == "answerable"]["answerable_em"].mean(), sub_df[sub_df["answerable"] == "unanswerable"]["answerable_em"].mean()
    answerable_f1, unanswerable_f1 = sub_df[sub_df["answerable"] == "answerable"]["answerable_f1"].mean(), sub_df[sub_df["answerable"] == "unanswerable"]["answerable_f1"].mean()
    total_answerable_acc = sub_df["answerable_acc"].mean()
    total_answerable_em = sub_df["answerable_em"].mean()
    total_answerable_f1 = sub_df["answerable_f1"].mean()

    hasanswer_ratio = df["hasanswer"].mean()
    df["hasanswer_acc"] = df.apply(lambda x: has_answer(x["answers"] if x["hasanswer"] else ['unanswerable'], x["pred"], tokenizer), axis=1)
    df["hasanswer_em"] = df.apply(lambda x: exact_match_score(x["pred"], x["answers"] if x["hasanswer"] else ['unanswerable']), axis=1)
    df["hasanswer_f1"] = df.apply(lambda x: f1_score(x["pred"], x["answers"] if x["hasanswer"] else ['unanswerable']), axis=1)
    hasanswer_acc, not_hasanswer_acc = df[df["hasanswer"]]["hasanswer_acc"].mean(), df[~df["hasanswer"]]["hasanswer_acc"].mean()
    hasanswer_em, not_hasanswer_em = df[df["hasanswer"]]["hasanswer_em"].mean(), df[~df["hasanswer"]]["hasanswer_em"].mean()
    hasanswer_f1, not_hasanswer_f1 = df[df["hasanswer"]]["hasanswer_f1"].mean(), df[~df["hasanswer"]]["hasanswer_f1"].mean()
    total_hasanswer_acc = df["hasanswer_acc"].mean()
    total_hasanswer_em = df["hasanswer_em"].mean()
    total_hasanswer_f1 = df["hasanswer_f1"].mean()
    metrics = {
        "Acc (hasanswer)": total_hasanswer_acc,
        "Ans Acc (hasanswer)": hasanswer_acc,
        "Unans Acc (hasanswer)": not_hasanswer_acc,
        "hasanswer_ratio": hasanswer_ratio,
        "answerable_ratio": answerable_ratio,
        "unanswerable_ratio": unanswerable_ratio,
        "uncertain_ratio": 1 - answerable_ratio - unanswerable_ratio,
        "Acc (answerable)": total_answerable_acc,
        "Ans Acc (answerable)": answerable_acc,
        "Unans Acc (answerable)": unanswerable_acc,
        "EM (hasanswer)": total_hasanswer_em,
        "Ans EM (hasanswer)": hasanswer_em,
        "Unans EM (hasanswer)": not_hasanswer_em,
        "F1 (hasanswer)": total_hasanswer_f1,
        "Ans EM (answerable)": answerable_em,
        "Unans EM (answerable)": unanswerable_em,
        "Em (answerable)": total_answerable_em,
        "F1 (answerable)": total_answerable_f1,
        }

    wandb.init(
        project=f"{args.project_prefix + '|' if args.project_prefix else ''}in-context-{args.task}", name=make_run_name(args), config=vars(args)
    )

    # metrics to dataframe
    df["answers"] = df["answers"].apply(lambda x: "||".join(x))
    df = df[["question", "answers", "prompt", "pred", "hasanswer", "hasanswer_acc", "answerable"]]
    wandb.log({"metrics": wandb.Table(dataframe=pd.DataFrame(data=metrics, index=[0]))})
    wandb.log({"raw output": wandb.Table(dataframe=df)})
    df["ctxs"] = dataset["ctxs"]
    df.to_csv(f"data/{make_run_name(args)}.csv", index=False)
    return df

def cal_conflict(dataset: Dataset, args):
    tokenizer = SimpleTokenizer()
    df = dataset.to_pandas()
    df = df.drop(columns=["ctxs"])
    df["is_valid_conflict_passage"] = dataset["is_valid_conflict_passage"]
    
    hasanswer_df = df[df.hasanswer]
    answerable_df = df[df.answerable == "answerable"]

    hasanswer_df["acc"] = hasanswer_df.apply(lambda x: has_answer(x["answers"] if not x["is_valid_conflict_passage"] else ["conflict"], x["pred"], SimpleTokenizer()), axis=1)
    answerable_df["acc"] = answerable_df.apply(lambda x: has_answer(x["answers"] if not x["is_valid_conflict_passage"] else ["conflict"], x["pred"], SimpleTokenizer()), axis=1)

    hasanswer_acc = hasanswer_df.acc.mean()
    hasanswer_non_conflict_acc = hasanswer_df[~hasanswer_df["is_valid_conflict_passage"]].acc.mean()
    hasanswer_conflict_acc = hasanswer_df[hasanswer_df["is_valid_conflict_passage"]].acc.mean()
    hasanswer_conflict_ratio = hasanswer_df[hasanswer_df["is_valid_conflict_passage"]].shape[0]/len(hasanswer_df)

    answerable_acc = answerable_df.acc.mean()
    answerable_non_conflict_acc = answerable_df[~answerable_df["is_valid_conflict_passage"]].acc.mean()
    answerable_conflict_acc = answerable_df[answerable_df["is_valid_conflict_passage"]].acc.mean()
    answerable_conflict_ratio = answerable_df[answerable_df["is_valid_conflict_passage"]].shape[0]/len(answerable_df)

    metrics = {
        "Acc (hasanswer)": hasanswer_acc,
        "Non-conflict (hasanswer)": hasanswer_non_conflict_acc,
        "Conflict (hasanswer)": hasanswer_conflict_acc,
        "Acc (answerable)": answerable_acc, 
        "Non-conflict (answerable)": answerable_non_conflict_acc,
        "Conflict (answerable)": answerable_conflict_acc,
        "hasanswer_conflict_ratio": hasanswer_conflict_ratio,
        "answerable_conflict_ratio": answerable_conflict_ratio,}

    wandb.init(
        project=f"{args.project_prefix + '|' if args.project_prefix else ''}in-context-{args.task}", name=make_run_name(args), config=vars(args)
    )
    df["answers"] = df["answers"].apply(lambda x: "||".join(x))
    df = df[["question", "answers", "prompt", "pred", "hasanswer", "answerable", "is_valid_conflict_passage"]]
    wandb.log({"metrics": wandb.Table(dataframe=pd.DataFrame(data=metrics, index=[0]))})
    wandb.log({"raw output": wandb.Table(dataframe=df)})
    df["ctxs"] = dataset["ctxs"]
    df.to_csv(f"data/{make_run_name(args)}.csv", index=False)
    return df