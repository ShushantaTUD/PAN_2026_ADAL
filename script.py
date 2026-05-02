"""
input file columns: Index(['id', 'text'], dtype='str')
Top five rows:
                                      id                                               text
0  50dd45c5-c310-5126-bb5d-f14d7a6d6958  'Ye was sayin', had I seen a lass wi' a lad's ...
1  1cb1e5cd-1267-5108-b5d8-f0acfabad5cb  We had a lovely time; certainly two of us had,...
2  d6aacddc-525b-5a40-baba-5b248625b0a1  “That store-bought stuff ain’t worth a hill o’...
3  44333da6-14de-5be4-ae28-8c3dd24f0ac0  The master of Nemours, to continue with the co...
4  35aa62c2-5159-53b9-9c43-186afdcd2a8a  Early in the month of October, 181-, Colonel S...
"""

import json
import argparse
import pandas as pd
import torch
# from typing import Dict, List, Optional, Tuple


from transformers import RobertaTokenizer, RobertaForSequenceClassification
# import torch

tokenizer = RobertaTokenizer.from_pretrained("Shushant/ADAL_AI_Detector")
model     = RobertaForSequenceClassification.from_pretrained("Shushant/ADAL_AI_Detector")

def predict(text: str) -> float:    
    model.eval()

    enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]

    return probs[1].item()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="script")
    parser.add_argument("-i", "--input", required=True, help="the input to the script.")
    parser.add_argument("-o", "--output", required=True, help="the output of the script.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(
        f"This is a demo, I ignore the passed input {args.input} and write some content into the output file"
        f" {args.output}."
    )

    input_full_path = args.input + "/dataset.jsonl"

    input_df = pd.read_json(input_full_path, lines=True)
    print(f"input file shape: {input_df.shape}")
    print(f"input file columns: {input_df.columns}")
    print(f"Top five rows:\n {input_df.head()}")

    pred_df = input_df.copy()
    pred_df["label"] = input_df["text"].apply(predict)

    pred_df[["id", "label"]].to_json(args.output + "/predictions.jsonl", orient="records", lines=True)
    print("Completed!!!")
