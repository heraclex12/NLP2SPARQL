#! /bin/bash
#
# Batch translation
#

trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

CODE=$DIR/nl2sparql
export PYTHONPATH="$CODE:$PYTHONPATH"

BERT_MODEL=$DIR/uncased_L-12_H-768_A-12
CORPUS=$DIR/corpus
DATA=$DIR/data
SRC=en
TRG=sparql

MODEL=model.stage2
TESTSET=test
BEAM=10
PENALTY=1.0
GPUID=
OUTPUT=

declare -A SUBSETS=(
    ["train"]="train"
    ["dev"]="valid"
    ["test"]="test"
)

#
# Usage
#
usage_exit () {
    echo "Usage $0 [-g GPUIDs] [-c TESTSET] output_prefix" 1>&2
    exit 1
}

#
# Options
#
while getopts s:t:g:c:h OPT; do
    case $OPT in
        s)  SRC=$OPTARG
            ;;
        t)  TRG=$OPTARG
            ;;
        g)  GPUID=$OPTARG
            ;;
        c)  TESTSET=$OPTARG
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done
shift $((OPTIND - 1))
if [ -n "$GPUID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPUID
fi
if [ $# -lt 1 ]; then
    usage_exit
else
    OUTPUT=$1
fi
SUBSET=${SUBSETS[$TESTSET]}

#
# Translation
#

test_main () {
    output=$1
    ### Translation
    fairseq-generate $DATA -s $SRC -t $TRG \
	--fp16 \
	--user-dir $CODE --task translation_with_bert \
	--bert-model $BERT_MODEL \
	--no-progress-bar \
	--gen-subset $SUBSET \
	--path $MODEL/checkpoint_best.pt \
	--lenpen $PENALTY --beam $BEAM --batch-size 32 \
	> $output.log

    ### Convert sub-words into words
    cat $output.log \
	| grep -e '^H\-' | sed -e 's/^H-//' \
	| sort -k 1 -n | cut -f 3 \
	| spm_decode --model=$CORPUS/train.spm.$TRG.model \
		     --input_format=piece > $output.out
}
test_main $OUTPUT
