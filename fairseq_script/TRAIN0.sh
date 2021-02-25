#! /bin/sh
#
# Tokenize raw sentences
#
trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

CODE=$DIR/nl2sparql
export PYTHONPATH="$CODE:$PYTHONPATH"

BERT_MODEL=bert-base-uncased
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
# Train sub-word models
#

### sentencepiece
sp_encode () {
    lang=$1
    size=$2
    spm_train --model_prefix=$CORPUS/train.spm.$lang \
	      --input=$CORPUS/train.$lang \
	      --vocab_size=$size \
	      --character_coverage=1.0 \
	      > $CORPUS/train.spm.$lang.log 2>&1
}

sp_encode   $TRG 4000 &
wait

#
# Apply the sub-word models
#

### sentencepiece
sp_decode () {
    lang=$1
    testset=$2
    cat $CORPUS/${testset}.$lang \
	| spm_encode --model=$CORPUS/train.spm.$lang.model \
	> $CORPUS/${testset}.bpe.$lang
}

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
    sp_decode   $TRG $testset &
    bert_decode $SRC $testset $BERT_MODEL &
    wait
done