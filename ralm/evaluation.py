from datasets import Dataset
import pandas as pd
import wandb
from utils import has_answer, SimpleTokenizer, make_run_name

def cal_unans(dataset: Dataset, args):
    tokenizer = SimpleTokenizer()
    df = dataset.to_pandas()
    df = df.drop(columns=["ctxs"])
    df["answers"] = df["answers"].apply(lambda x: "||".join(x))
    answerable_ratio = df["answerable"].value_counts(normalize=True).get("answerable", 0)
    df["answerable_acc"] = df.apply(lambda x: has_answer(x["answers"] if x["answerable"] == "answerable" else ['unanswerable'], x["pred"], tokenizer), axis=1)
    answerable_acc, unanswerable_acc = df[df["answerable"] == "answerable"]["answerable_acc"].mean(), df[df["answerable"] == "unanswerable"]["answerable_acc"].mean()
    total_answerable_acc = df["answerable_acc"].mean()
    metrics = {
        "Acc": total_answerable_acc,
        "Acc (ans)": answerable_acc,
        "Acc (unans)": unanswerable_acc,
        "answerable_ratio": answerable_ratio,
    }
    if not args.filter_uncertain:
        answerable_ratio = df["hasanswer"].mean()
        df["hasanswer_acc"] = df.apply(lambda x: has_answer(x["answers"] if x["hasanswer"] else ['unanswerable'], x["pred"], tokenizer), axis=1)
        answerable_acc, unanswerable_acc = df[df["hasanswer"]]["hasanswer_acc"].mean(), df[~df["hasanswer"]]["hasanswer_acc"].mean()
        total_answerable_acc = df["hasanswer_acc"].mean()
        metrics.update({
            "Acc (hasans)": total_answerable_acc,
            "Acc (hasans, ans)": answerable_acc,
            "Acc (hasans, unans)": unanswerable_acc,
            "hasanswer_ratio": answerable_ratio,
        })
    wandb.init(
        project=f"in-context-{args.task}", name=make_run_name(args), config=vars(args)
    )

    # metrics to dataframe
    df = df[["question", "answers", "prompt", "pred", "hasanswer", "answerable", "answerable_acc"]]
    wandb.log({"metrics": wandb.Table(dataframe=pd.DataFrame(data=metrics, index=[0]))})
    wandb.log({"raw output": wandb.Table(dataframe=df)})
    df["ctxs"] = dataset["ctxs"]
    df.to_csv(f"data/{make_run_name(args)}.csv", index=False)
    return df