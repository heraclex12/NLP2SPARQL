"""
Create BERT Vocabulary
"""

import sys
import argparse
from transformers import BertTokenizer

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    for i, (tok, freq) in enumerate(tokenizer.vocab.items()):
      if i < 5: continue
      print(f"{tok} {freq}")

def cli_main():
    parser = argparse.ArgumentParser(description="Create BERT vocabulary")
    parser.add_argument("-m", "--model", type=str, metavar='DIR', dest="bert_model",
                        required=True, help="path to the BERT model")
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    cli_main()
