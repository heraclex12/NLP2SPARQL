import re
import argparse
from generator_utils import encode

def preprocess_sentence(w):
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip().lower()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w

def preprocess_sparql(s):

    # Remove prefixes
    s = re.sub(r"PREFIX\s[^\s]*\s[^\s]*", "", s)
    s = encode(s.rstrip().strip())

    s = ' '.join(s.split())

    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--en', dest='english_file', required=True)
    parser.add_argument('--sparql', dest='sparql_file', required=True)
    parser.add_argument('--output', dest='output_prefix', required=True)
    args = parser.parse_args()

    english_file = args.english_file
    sparql_file = args.sparql_file
    output_prefix = args.output_prefix

    with open(output_prefix+'.en', 'w') as out: 
        with open(english_file) as f:
            for line in f:
                if '\n' == line[-1]:
                    line = line[:-1]
                out.write(preprocess_sentence(line))
                out.write('\n')
    
    with open(output_prefix+'.sparql', 'w') as out: 
        with open(sparql_file) as f:
            for line in f:
                if '\n' == line[-1]:
                    line = line[:-1]
                out.write(preprocess_sparql(line))
                out.write('\n')