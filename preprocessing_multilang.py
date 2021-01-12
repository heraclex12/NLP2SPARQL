from preprocessing import preprocess_sparql, preprocess_sentence
import argparse
import os
import re

QALD_LANGUAGES = {'de', 'ru', 'pt', 'en', 'hi_IN', 'fa', 'pt_BR', 'it', 'fr', 'ro', 'es', 'nl'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data_dir', required=True)
    parser.add_argument('--subset', dest='subset', required=True)
    parser.add_argument('--output', dest='output_dir', required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    subset = args.subset
    output_dir = args.output_dir

    data_dir = data_dir.rstrip('/')
    output_dir = output_dir.rstrip('/')

    for lang in QALD_LANGUAGES:
        if not os.path.exists(f'{output_dir}/{lang}/'):
            os.mkdir(f'{output_dir}/{lang}/')

        with open(f'{output_dir}/{lang}/{subset}.lang', 'w') as out:
            with open(f'{data_dir}/{lang}/{subset}.lang', 'r') as f:
                for line in f:
                    if '\n' == line[-1]:
                        line = line[:-1]
                    out.write(preprocess_sentence(line))
                    out.write('\n')

        with open(f'{output_dir}/{lang}/{subset}.sparql', 'w') as out:
            with open(f'{data_dir}/{lang}/{subset}.sparql', 'r') as f:
                for line in f:
                    if '\n' == line[-1]:
                        line = line[:-1]
                    out.write(preprocess_sparql(line))
                    out.write('\n')