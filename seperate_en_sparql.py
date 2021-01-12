import json

if __name__ == '__main__':
    en_file = open('VQUANDA/test.en', 'w', encoding='latin1')
    sparql_file = open('VQUANDA/test.sparql', 'w', encoding='latin1')
    data = json.load(open('VQUANDA/test.json', 'r', encoding='latin1'))
    for element in data:
        en_file.write(element['question'] + '\n')
        sparql_file.write(element['query'] + '\n')

    en_file.close()
    sparql_file.close()