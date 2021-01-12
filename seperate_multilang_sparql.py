import json
import argparse
import os

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

    data = json.load(open(f'{data_dir}/{subset}.json', 'r'))
    for lang in QALD_LANGUAGES:
        if not os.path.exists(f'{output_dir}/{lang}/'):
            os.mkdir(f'{output_dir}/{lang}/')
        sparql_file = open(f'{output_dir}/{lang}/{subset}.sparql', 'w')
        language_file = open(f'{output_dir}/{lang}/{subset}.lang', 'w')
        for element in data['questions']:
            for question in element['question']:
                if question['language'] == lang and 'string' in question:
                    language_file.write(question['string'] + '\n')
                    sparql_file.write(element['query']['sparql'] + '\n')
                    break

        language_file.close()
        sparql_file.close()
