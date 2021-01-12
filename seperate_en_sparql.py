import json
import argparse

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

    en_file = open(f'{output_dir}/{subset}.en', 'w')
    sparql_file = open(f'{output_dir}/{subset}.sparql', 'w')
    data = json.load(open(f'{data_dir}/{subset}.json', 'r'))
    for element in data:
        en_file.write(element['question'] + '\n')
        sparql_file.write(element['query'] + '\n')

    en_file.close()
    sparql_file.close()
