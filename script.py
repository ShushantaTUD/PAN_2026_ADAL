# """
# input file columns: Index(['id', 'text'], dtype='str')
# Top five rows:
#                                       id                                               text
# 0  50dd45c5-c310-5126-bb5d-f14d7a6d6958  'Ye was sayin', had I seen a lass wi' a lad's ...
# 1  1cb1e5cd-1267-5108-b5d8-f0acfabad5cb  We had a lovely time; certainly two of us had,...
# 2  d6aacddc-525b-5a40-baba-5b248625b0a1  “That store-bought stuff ain’t worth a hill o’...
# 3  44333da6-14de-5be4-ae28-8c3dd24f0ac0  The master of Nemours, to continue with the co...
# 4  35aa62c2-5159-53b9-9c43-186afdcd2a8a  Early in the month of October, 181-, Colonel S...
# """

# import json
# import argparse
# import pandas as pd
# import torch
# # from typing import Dict, List, Optional, Tuple


# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# # import torch

# def predict(text: str) -> float:

#     tokenizer = RobertaTokenizer.from_pretrained("Shushant/ADAL_AI_Detector")
#     model     = RobertaForSequenceClassification.from_pretrained("Shushant/ADAL_AI_Detector")
#     model.eval()

#     enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         probs = torch.softmax(model(**enc).logits, dim=-1)[0]

#     return probs[1].item()



# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(prog="script")
#     parser.add_argument("-i", "--input", required=True, help="the input to the script.")
#     parser.add_argument("-o", "--output", required=True, help="the output of the script.")

#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     print(
#         f"This is a demo, I ignore the passed input {args.input} and write some content into the output file"
#         f" {args.output}."
#     )

#     input_full_path = args.input + "/dataset.jsonl"

#     input_df = pd.read_json(input_full_path, lines=True)
#     print(f"input file shape: {input_df.shape}")
#     print(f"input file columns: {input_df.columns}")
#     print(f"Top five rows:\n {input_df.head()}")

#     pred_df = input_df.copy()
#     pred_df["label"] = input_df["text"].apply(predict)

#     pred_df[["id", "label"]].to_json(args.output + "/predictions.jsonl", orient="records", lines=True)
#     print("Completed!!!")



"""
PAN'26 Voight-Kampff Generative AI Detection — ADAL inference

Invocation (per task spec):
    python3 script.py -i $inputDataset/dataset.jsonl -o $outputDir

Input:  a single JSONL file, each line: {"id": "...", "text": "..."}
Output: $outputDir/predictions.jsonl, each line: {"id": "...", "label": <float in [0,1]>}
        label > 0.5 -> machine-generated, < 0.5 -> human, == 0.5 -> undecidable
"""
import os
import json
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local path inside the image — populated at build time (see Dockerfile).
# Must NOT hit the network at inference time (TIRA is sandboxed).
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

breakpoint()

print(f"[adal] Loading model from {MODEL_DIR} on {DEVICE} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(DEVICE).eval()


@torch.no_grad()
def predict_batch(texts, batch_size=16, max_length=512):
    """Return a list of P(machine-generated) for each text. Processes in isolation per task rules."""
    probs_out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(DEVICE)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]   # column 1 = machine
        probs_out.extend(probs.detach().cpu().tolist())
    return probs_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="adal-inference")
    p.add_argument("-i", "--input", required=True,
                   help="Absolute path to the input JSONL file (dataset.jsonl).")
    p.add_argument("-o", "--output", required=True,
                   help="Absolute path to the output directory.")
    return p.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PAN'26 passes the file path directly. Be tolerant if a directory is passed too.
    if in_path.is_dir():
        candidate = in_path / "dataset.jsonl"
        if candidate.exists():
            in_path = candidate
        else:
            raise FileNotFoundError(f"No dataset.jsonl in directory {in_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[adal] Reading {in_path}", flush=True)
    df = pd.read_json(in_path, lines=True)
    print(f"[adal] {len(df)} rows, columns: {list(df.columns)}", flush=True)

    texts = df["text"].astype(str).tolist()
    scores = predict_batch(texts)

    out_path = out_dir / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for _id, score in zip(df["id"].tolist(), scores):
            f.write(json.dumps({"id": _id, "label": float(score)}) + "\n")

    print(f"[adal] Wrote {len(scores)} predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()