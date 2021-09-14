
<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- PROJECT LOGO -->
<br />
<h1>SPBERT: A Pre-trained Model for SPARQL Query Language </h1>
    <br />

<!-- ABOUT THE PROJECT -->
In this project, we provide the code for reproducing the experiments in [our paper](https://arxiv.org/abs/2106.09997). SPBERT is a BERT-based language model pre-trained on massive SPARQL query logs. SPBERT can learn general-purpose representations in both natural language and SPARQL query language and make the most of the sequential order of words that are crucial for structured language like SPARQL.

### Prerequisites

To reproduce the experiment of our model, please install the requirements.txt according to the following instructions:
* transformers==4.5.1
* pytorch==1.8.1
* python 3.7.10
```sh
$ pip install -r requirements.txt
```

### Pre-trained models
We release three versions of pre-trained weights. Pre-training was based on the [original BERT code](https://github.com/google-research/bert) provided by Google, and training details are described in our paper. You can download all versions from the table below:
| Pre-training objective | Model | Steps | Link |
|---|:---:|:---:|:---:|
| MLM  | SPBERT (scratch) | 200k | 🤗 [razent/spbert-mlm-zero](https://huggingface.co/razent/spbert-mlm-zero) |
| MLM  | SPBERT (BERT-initialized) | 200k | 🤗 [razent/spbert-mlm-base](https://huggingface.co/razent/spbert-mlm-base) |
| MLM+WSO  | SPBERT (BERT-initialized) | 200k | 🤗 [razent/spbert-mlm-wso-base](https://huggingface.co/razent/spbert-mlm-wso-base) |

### Datasets
All evaluation datasets can download [here](https://drive.google.com/drive/folders/1m_pJ0prUDpCWAFuxlvp_S48hGG_AASjb?usp=sharing).

### Example
To fine-tune models:
```bash
python run.py \
        --do_train \
        --do_eval \
        --model_type bert \
        --model_architecture bert2bert \
        --encoder_model_name_or_path bert-base-cased \
        --decoder_model_name_or_path sparql-mlm-zero \
        --source sparql \
        --target en \
        --train_filename ./LCQUAD/train \
        --dev_filename ./LCQUAD/dev \
        --output_dir ./ \
        --max_source_length 64 \
        --weight_decay 0.01 \
        --max_target_length 128 \
        --beam_size 10 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --save_inverval 10 \
        --num_train_epochs 150
```

To evaluate models:
```bash
python run.py \
        --do_test \
        --model_type bert \
        --model_architecture bert2bert \
        --encoder_model_name_or_path bert-base-cased \
        --decoder_model_name_or_path sparql-mlm-zero \
        --source sparql \
        --target en \
        --load_model_path ./checkpoint-best-bleu/pytorch_model.bin \
        --dev_filename ./LCQUAD/dev \
        --test_filename ./LCQUAD/test \
        --output_dir ./ \
        --max_source_length 64 \
        --max_target_length 128 \
        --beam_size 10 \
        --eval_batch_size 32 \
```

<!-- CONTACT -->
## Contact
Email: [heraclex12@gmail.com](mailto:heraclex12@gmail.com) - Hieu Tran


## Citation
```
@misc{tran2021spbert,
      title={SPBERT: An Efficient Pre-training BERT on SPARQL Queries for Question Answering over Knowledge Graphs}, 
      author={Hieu Tran and Long Phan and James Anibal and Binh T. Nguyen and Truong-Son Nguyen},
      year={2021},
      eprint={2106.09997},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

