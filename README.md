
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
In this project, we provide the code for reproducing the experiments in [our paper](#). SPBERT is a BERT-based language model pre-trained on massive SPARQL query logs. SPBERT can learn general-purpose representations in both natural language and SPARQL query language and make the most of the sequential order of words that are crucial for structured language like SPARQL.

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
| MLM  | SPBERT (scratch) | 200k | [Here](#) |
| MLM  | SPBERT (BERT-initialized) | 200k | [Here](#) |
| MLM+WSO  | SPBERT (BERT-initialized) | 200k | [Here](#) |

### Datasets


<!-- CONTACT -->
## Contact
Coming Soon


## Citation
Coming Soon

