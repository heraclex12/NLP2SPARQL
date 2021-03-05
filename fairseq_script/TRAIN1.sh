#! /bin/sh
#
# Make binalized dataset
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

#
# Usage
#
usage_exit () {
    echo "Usage $0 [-s SRC] [-t TRG]" 1>&2
    exit 1
}

#
# Options
#
while getopts s:t:h OPT; do
    case $OPT in
        s)  SRC=$OPTARG
            ;;
        t)  TRG=$OPTARG
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done
shift $((OPTIND - 1))

#
# Main
#

mkdir -p $DATA

### Convert the BERT vocabulary into fairseq one.
cat $BERT_MODEL/vocab.txt \
    | tail -n +5 \
    | sed -e 's/$/ 0/' \
	  > $DATA/dict.$SRC.txt

### Encode corpora into binary sets.
python $CODE/preprocess.py \
    --workers 4 \
    --source-lang $SRC --target-lang $TRG \
    --srcdict $DATA/dict.$SRC.txt \
    --trainpref $CORPUS/train.bpe \
    --validpref $CORPUS/dev.bpe \
    --testpref $CORPUS/test.bpe \
    --destdir $DATA
