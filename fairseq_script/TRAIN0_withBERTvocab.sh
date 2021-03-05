#! /bin/sh
#
# Tokenize raw sentences
#
trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

CODE=$DIR/nl2sparql
export PYTHONPATH="$CODE:$PYTHONPATH"

BERT_MODEL=$DIR/cased_L-12_H-768_A-12
CORPUS=$DIR/corpus
SRC=en
TRG=sparql

#
# Download the corpus
#
mkdir -p $CORPUS
cp ../../drive/MyDrive/THESIS/tntspa/data/LC-QUAD/train.* $CORPUS
cp ../../drive/MyDrive/THESIS/tntspa/data/LC-QUAD/test.* $CORPUS
cp ../../drive/MyDrive/THESIS/tntspa/data/LC-QUAD/dev.* $CORPUS

#
# Download the BERT model
#
if [ ! -d $BERT_MODEL ]; then
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    unzip cased_L-12_H-768_A-12.zip
    (cd $BERT_MODEL ; \
     ln -s bert_config.json config.json ; \
     transformers-cli convert --model_type bert \
	--tf_checkpoint bert_model.ckpt \
	--config bert_config.json \
	--pytorch_dump_output pytorch_model.bin)
fi



### BERT tokenizer
bert_decode () {
    lang=$1
    testset=$2
    model=$3
    cat $CORPUS/${testset}.$lang \
	| python $CODE/bert_tokenize.py \
		  --model=$model \
		  > $CORPUS/${testset}.bpe.$lang
}

for testset in train test dev; do
    bert_decode $TRG $testset $BERT_MODEL &
    bert_decode $SRC $testset $BERT_MODEL &
    wait
done
