import collections
import http.client
import json
import logging
import re
import sys
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
from functools import reduce

ENDPOINT = "http://dbpedia.org/sparql"
GRAPH = "http://dbpedia.org"


def log_statistics(used_resources, special_classes, not_instanced_templates):
    total_number_of_resources = len(used_resources)
    total_number_of_filled_placeholder_positions = sum(used_resources.values())
    examples_per_instance = collections.Counter()
    for resource in used_resources:
        count = used_resources[resource]
        examples_per_instance.update([count])

    logging.info('{:6d} used resources in {} placeholder positions'.format(total_number_of_resources,
                                                                           total_number_of_filled_placeholder_positions))
    for usage in examples_per_instance:
        logging.info('{:6d} resources occur \t{:6d} times \t({:6.2f} %) '.format(examples_per_instance[usage], usage,
                                                                                 examples_per_instance[
                                                                                     usage] * 100 / total_number_of_resources))
    for cl in special_classes:
        logging.info('{} contains: {}'.format(cl, ', '.join(special_classes[cl])))
    logging.info('{:6d} not instanciated templates:'.format(sum(not_instanced_templates.values())))
    for template in not_instanced_templates:
        logging.info('{}'.format(template))


def save_cache(file, cache):
    ordered = collections.OrderedDict(cache.most_common())
    with open(file, 'w') as outfile:
        json.dump(ordered, outfile)


def query_dbpedia(query):
    param = dict()
    param["default-graph-uri"] = GRAPH
    param["query"] = query
    param["format"] = "JSON"
    param["CXML_redir_for_subjs"] = "121"
    param["CXML_redir_for_hrefs"] = ""
    param["timeout"] = "600"  # ten minutes - works with Virtuoso endpoints
    param["debug"] = "on"
    try:
        resp = urllib.request.urlopen(ENDPOINT + "?" + urllib.parse.urlencode(param))
        j = resp.read()
        resp.close()
    except (urllib.error.HTTPError, http.client.BadStatusLine):
        logging.debug("*** Query error. Empty result set. ***")
        j = '{ "results": { "bindings": [] } }'
    sys.stdout.flush()
    return json.loads(j)


def strip_brackets(s):
    s = re.sub(r'\([^)]*\)', '', s)
    if "," in s:
        s = s[:s.index(",")]
    return s.strip().lower()


SPARQL_KEYWORDS = {
    'SELECT', 'CONSTRUCT', 'ASK', 'DESCRIBE', 'BIND', 'WHERE', 'LIMIT',
    'VALUES', 'DISTINCT', 'AS', 'FILTER', 'ORDER', 'BY', 'SERVICE', 'OFFSET',
    'NOT', 'EXISTS', 'OPTIONAL', 'UNION', 'FROM', 'GRAPH', 'NAMED', 'DESC',
    'ASC', 'REDUCED', 'STR', 'LANG', 'LANGMATCHES', 'REGEX', 'BOUND', 'DATATYPE',
    'ISBLANK', 'ISLITERAL', 'ISIRI', 'ISURI', 'GROUP_CONCAT', 'GROUP', 'DELETE', 'CLEAR',
    'CREATE', 'COPY', 'DROP', 'INSERT', 'LOAD', 'DATA', 'INTO', 'WITH', 'ALL', 'SILENT',
    'DEFAULT', 'USING', 'MD5', 'SHA1', 'SHA256', 'SHA384', 'SHA512', 'STRSTARTS',
    'STRENDS', 'SAMETERM', 'ISNUMERIC', 'UCASE', 'SUBSTR', 'STRLEN', 'STRBEFORE', 'STRAFTER',
    'REPLACE', 'LEVENSHTEIN_DIST', 'LCASE', 'ENCODE_FOR_URI', 'CONTAINS', 'CONCAT',
    'COALESCE', 'CHOOSE_BY_MAX', 'CHOOSE_BY_MIN', 'YEAR', 'DAY', 'TZ', 'TIMEZONE', 'HOURS',
    'MINUTES', 'MONTH', 'NOW', 'DUR_TO_USECS', 'SECONDS_DBL', 'USECS_TO_DUR', 'IF', 'MINUS',
    'AVG', 'COUNT', 'MAX', 'MIN', 'SAMPLE', 'SUM', 'ABS', 'ADD', 'BASE', 'CEIL', 'COS', 'FLOOR',
    'HAMMING_DIST', 'HAVERSINE_DIST', 'LN', 'LOG2', 'MOD', 'POWER', 'RADIANS', 'RAND',
    'ROUND', 'ROUNDDOWN', 'ROUNDUP', 'TAN', 'VAR', 'VARP'
}

REPLACEMENTS = [
    ['dbo:', 'onto:', 'http://dbpedia.org/ontology/', 'dbo_'],
    ['dbp:', 'http://dbpedia.org/property/', 'dbp_'],
    ['dbc:', 'http://dbpedia.org/resource/Category:', 'dbc_'],
    ['dbr:', 'res:', 'http://dbpedia.org/resource/', 'dbr_'],
    ['rdf:', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'rdf_'],
    ['rdfs:', 'http://www.w3.org/2000/01/rdf-schema#', 'rdfs_'],
    ['xsd:', 'http://www.w3.org/2001/XMLSchema#', 'xsd_'],
    ['dct:', 'http://purl.org/dc/terms/', 'dct_'],
    ['dc:', 'http://purl.org/dc/elements/1.1/', 'dc_'],
    ['georss:', 'http://www.georss.org/georss/', 'georss_'],
    ['geo:', 'http://www.opengis.net/ont/geosparql#', 'geo_'],
    ['geof:', 'http://www.opengis.net/def/function/geosparql/', 'geof_'],
    ['vrank:', 'http://purl.org/voc/vrank#', 'vrank_'],
    ['bif:', 'bif_'],
    ['foaf:', 'http://xmlns.com/foaf/0.1/', 'foaf_'],
    ['owl:', 'http://www.w3.org/2002/07/owl#', 'owl_'],
    ['yago:', 'http://dbpedia.org/class/yago/', 'yago_'],
    ['skos:', 'http://www.w3.org/2004/02/skos/core#', 'skos_'],
    ['?', 'var_'],
    ['*', ' wildcard '],
    [' <= ', ' math_leq '],
    [' >= ', ' math_geq '],
    [' != ', ' math_neq '],
    [' = ', ' math_eql '],
    [' < ', ' math_lt '],
    [' > ', ' math_gt '],
    [' ; ', ' sep_semi '],
    ['; ', ' sep_semi '],
    ['"', " quote_str "],
    [', ', ' sep_com '],
    [' , ', ' sep_com '],
    ['^^', ' str_type '],
    ['||', ' or_logical '],
    ['&&', ' and_logical '],
    [' ! ', ' bool_not '],
    ['@', ' lang_at '],
    [' ( ', '  par_open  '],
    [' ) ', '  par_close  '],
    ['(', ' attr_open '],
    [')', ') ', ' attr_close '],
    ['{', ' brack_open '],
    ['}', ' brack_close '],
    [' . ', ' sep_dot '],
    ['. ', ' sep_dot '],
]

STANDARDS = {
    'dbo_almaMater': ['dbp_almaMater'],
    'dbo_award': ['dbp_awards'],
    'dbo_birthPlace': ['dbp_birthPlace', 'dbp_placeOfBirth', 'dbp_birthplace'],
    'dbo_deathPlace': ['dbp_deathPlace', 'dbp_placeOfDeath'],
    'dbo_child': ['dbp_children'],
    'dbo_college': ['dbp_college'],
    'dbo_hometown': ['dbp_hometown', 'dbp_homeTown'],
    'dbo_nationality': ['dbo_stateOfOrigin', 'dbp_nationality'],
    'dbo_relative': ['dbp_relatives'],
    'dbo_restingPlace': ['dbp_restingPlaces', 'dbp_placeOfBurial', 'dbp_placeofburial', 'dbo_placeOfBurial',
                         'dbp_restingplace', 'dbp_restingPlace'],
    'dbo_spouse': ['dbp_spouse'],
    'dbo_partner': ['dbp_partner'],
    'dbo_parent': ['dbp_parent'],
    'dbo_lyrics': ['dbp_lyrics'],
    'dbo_ground': ['dbp_ground'],
    'dbo_owner': ['dbp_owner', 'dbp_owners'],
    'dbo_species': ['dbp_species'],
    'dbo_series': ['dbp_series'],
    'dbo_commander': ['dbp_commander'],
    'dbo_majorShrine': ['dbp_majorShrine'],
    'dbo_firstDriver': ['dbp_firstDriver'],
    'dbo_ethnicity': ['dbp_ethnicity'],
    'dbo_sourceCountry': ['dbp_sourceCountry'],
    'dbo_publisher': ['dbp_publisher'],
    'dbo_field': ['dbp_field', 'dbp_fields'],
    'dbo_governmentType': ['dbp_governmentType'],
    'dbo_programmingLanguage': ['dbp_programmingLanguage'],
    'dbo_gender': ['dbp_gender'],
    'dbo_operatingSystem': ['dbp_operatingSystem'],
    'dbo_editing': ['dbp_editing'],
    'dbo_artist': ['dbp_artist'],
    'dbo_location': ['dbp_location', 'dbp_locations'],
    'dbo_family': ['dbp_family'],
    'dbo_format': ['dbp_format'],
    'dbo_leader': ['dbp_leader'],
    'dbo_class': ['dbp_class'],
    'dbo_composer': ['dbp_composer'],
    'dbo_builder': ['dbp_builder'],
    'dbo_archipelago': ['dbp_archipelago'],
    'dbo_origin': ['dbp_origin'],
    'dbo_type': ['dbp_type'],
    'dbo_successor': ['dbp_successor'],
    'dbo_architecturalStyle': ['dbp_architecturalStyle'],
    'dbo_stadium': ['dbp_stadium'],
    'dbo_billed': ['dbp_billed'],
    'dbo_border': ['dbp_border'],
    'dbo_parentCompany': ['dbp_parentCompany'],
    'dbo_lieutenant': ['dbp_lieutenant'],
    'dbo_club': ['dbp_club'],
    'dbo_starring': ['dbp_starring'],
    'dbo_animator': ['dbp_animator'],
    'dbo_athletics': ['dbp_athletics'],
    'dbo_executiveProducer': ['dbp_executiveProducer'],
    'dbo_presenter': ['dbp_presenter'],
    'dbo_developer': ['dbp_developer'],
    'dbo_manager': ['dbp_manager'],
    'dbo_channel': ['dbp_channel'],
    'dbo_creator': ['dbp_creator', 'dbp_creators'],
    'dbo_poleDriver': ['dbp_poleDriver'],
    'dbo_chairman': ['dbp_chairman'],
    'dbo_appointer': ['dbp_appointer'],
    'dbo_founder': ['dbp_founder'],
    'dbo_draftTeam': ['dbp_draftTeam'],
    'dbo_bodyDiscovered': ['dbp_bodyDiscovered'],
    'dbo_instrument': ['dbp_instrument', 'dbp_instruments'],
    'dbo_league': ['dbp_league'],
    'dbo_champion': ['dbp_champion'],
    'dbo_governor': ['dbp_governor'],
    'dbo_genre': ['dbp_genre'],
    'dbo_incumbent': ['dbp_incumbent'],
    'dbo_illustrator': ['dbp_illustrator'],
    'dbo_school': ['dbp_school'],
    'dbo_doctoralAdvisor': ['dbp_doctoralAdvisor'],
    'dbo_deputy': ['dbp_deputy'],
    'dbo_phylum': ['dbp_phylum'],
    'dbo_state': ['dbp_state'],
    'dbo_veneratedIn': ['dbp_veneratedIn'],
    'dbo_operator': ['dbp_operator'],
    'dbo_institution': ['dbp_institution'],
    'dbo_opponent': ['dbp_opponent'],
    'dbo_jurisdiction': ['dbp_jurisdiction'],
    'dbo_river': ['dbp_river'],
    'dbo_education': ['dbp_education'],
    'dbo_assembly': ['dbp_assembly'],
    'dbo_sire': ['dbp_sire'],
    'dbo_sourceRegion': ['dbp_sourceRegion'],
    'dbo_author': ['dbp_author'],
    'dbo_coverArtist': ['dbp_coverArtist'],
    'dbo_affiliation': ['dbp_affiliation', 'dbp_affiliations'],
    'dbo_honours': ['dbp_honours'],
    'dbo_religion': ['dbp_religion'],
    'dbo_currency': ['dbp_currency'],
    'dbo_inflow': ['dbp_inflow'],
    'dbo_country': ['dbp_country'],
    'dbo_city': ['dbp_city', 'dbp_cities'],
    'dbo_party': ['dbp_party'],
    'dbo_homeStadium': ['dbp_homeStadium'],
    'dbo_network': ['dbp_network'],
    'dbo_mayor': ['dbp_mayor'],
    'dbo_director': ['dbp_director'],
    'dbo_manufacturer': ['dbp_manufacturer'],
    'dbo_university': ['dbp_university'],
    'dbo_locationCity': ['dbp_locationCity'],
    'dbo_trainer': ['dbp_trainer'],
    'dbo_citizenship': ['dbp_citizenship'],
    'dbo_architect': ['dbp_architect'],
    'dbo_locationCountry': ['dbp_locationCountry'],
    'dbo_grandsire': ['dbp_grandsire'],
    'dbo_largestCity': ['dbp_largestCity'],
    'dbo_employer': ['dbp_employer'],
    'dbo_related': ['dbp_related'],
    'dbo_producer': ['dbp_producer'],
    'dbo_designer': ['dbp_designer'],
    'dbo_race': ['dbp_race'],
    'dbo_discoverer': ['dbp_discoverer'],
    'dbo_garrison': ['dbp_garrison'],
    'dbo_cinematography': ['dbp_cinematography'],
    'dbo_highschool': ['dbp_highschool', 'dbp_highSchool'],
    'dbo_place': ['dbp_place'],
    'dbo_industry': ['dbp_industry'],
    'dbo_canonizedBy': ['dbp_canonizedBy'],
    'dbo_writer': ['dbp_writer', 'dbp_writers'],
    'dbo_district': ['dbp_district'],
    'dbo_crosses': ['dbp_crosses'],
    'dbo_denomination': ['dbp_denomination'],
    'dbo_narrator': ['dbp_narrator'],
    'dbo_company': ['dbp_company'],
    'dbo_commandStructure': ['dbp_commandStructure'],
    'dbo_license': ['dbp_license'],
    'dbo_leaderName': ['dbp_leaderName'],
    'dbo_team': ['dbp_team'],
    'dbo_rival': ['dbp_rival'],
    'dbo_influencedBy': ['dbp_influencedBy'],
    'dbo_capital': ['dbp_capital'],
    'dbo_training': ['dbp_training'],
    'dbo_album': ['dbp_album'],
    'dbo_binomialAuthority': ['dbp_binomialAuthority'],
    'dbo_placeOfBurial': ['dbp_placeOfBurial'],
    'dbo_distributor': ['dbp_distributor'],
    'dbo_predecessor': ['dbp_predecessor'],
    'dbo_monarch': ['dbp_monarch'],
    'dbo_prospectTeam': ['dbp_prospectTeam'],
    'dbo_residence': ['dbp_residence'],
    'dbo_occupation': ['dbp_occupation'],
    'dbo_portrayer': ['dbp_portrayer'],
    'dbo_governingBody': ['dbp_governingBody'],
    'dbo_nearestCity': ['dbp_nearestCity'],
    'dbo_deathCause': ['dbp_deathCause'],
    'dbo_position': ['dbp_position'],
    'dbo_language': ['dbp_language', 'dbp_languages'],
    'dbo_county': ['dbp_county', 'dbp_counties'],
    'dbo_order': ['dbp_order'],
    'dbo_broadcastArea': ['dbp_broadcastArea'],
    'dbo_coach': ['dbp_coach'],
    'dbo_mouthCountry': ['dbp_mouthCountry'],
    'dbo_outflow': ['dbp_outflow'],
    'dbo_superintendent': ['dbp_superintendent'],
    'dbo_knownFor': ['dbp_knownFor'],
    'dbo_president': ['dbp_president'],
    'dbo_movement': ['dbp_movement'],
    'dbo_region': ['dbp_region'],
    'dbo_editor': ['dbp_editor'],
    'dbo_voice': ['dbp_voices'],
    'dbp_title': ['dbp_titles'],
    'dbo_tenant': ['dbp_tenants'],
    'dbo_stylisticOrigin': ['dbp_stylisticOrigins'],
    'dbo_sisterStation': ['dbp_sisterStations'],
    'dbo_service': ['dbp_services'],
    'dbo_product': ['dbp_products'],
    'dbo_primeMinister': ['dbp_primeminister'],
    'dbo_notablework': ['dbp_notableworks'],
    'dbo_notableCommander': ['dbp_notableCommanders'],
    'dbo_neighboringMunicipality': ['dbp_neighboringMunicipalities'],
    'dbo_managerClub': ['dbp_managerclubs'],
    'dbo_mainInterest': ['dbp_mainInterests'],
    'dbo_house': ['dbp_houses'],
    'dbo_headquarter': ['dbp_headquarters'],
    'dbp_headcoach': ['dbp_headCoach'],
    'dbo_formerCoach': ['dbp_formercoach'],
    'dbo_doctoralStudent': ['dbp_doctoralStudents'],
    'dbo_division': ['dbp_divisions'],
    'dbo_destination': ['dbp_destinations'],
    'dbo_debutTeam': ['dbp_debutteam'],
    'dbp_currentTeam': ['dbp_currentteam'],
    'dbo_currentMember': ['dbp_currentMembers'],
    'dbo_battle': ['dbp_battles'],
    'dbo_openingTheme': ['dbp_opentheme']
}


def encode(sparql):
    encoded_sparql = do_replacements(sparql)
    shorter_encoded_sparql = shorten_query(encoded_sparql)
    normalized = normalize_predicates(shorter_encoded_sparql)
    return normalized


def decode(encoded_sparql):
    short_sparql = reverse_replacements(encoded_sparql)
    sparql = reverse_shorten_query(short_sparql)
    return ' '.join(sparql.split())


def normalize_predicates(sparql):
    sparql = re.sub(r"dbo_([A-Z])([a-zA-Z]+)", lambda matches: 'dbo_' + matches[1].lower() + matches[2], sparql)
    for standard in STANDARDS:
        for alternative in STANDARDS[standard]:
            sparql = re.sub(f'<{alternative}>', f'<{standard}>', sparql)
            # sparql = sparql.replace(alternative, standard)

    return sparql


def do_replacements(sparql):
    for r in REPLACEMENTS:
        encoding = r[-1]
        for original in r[:-1]:
            sparql = sparql.replace(original, encoding)
    return sparql


def reverse_replacements(sparql):
    for r in REPLACEMENTS:
        original = r[0]
        encoding = r[-1]
        sparql = sparql.replace(encoding, original)
        stripped_encoding = str.strip(encoding)
        sparql = sparql.replace(stripped_encoding, original)
        sparql = sparql.replace('{', ' { ').replace('}', ' } ')
    return sparql


def shorten_query(sparql):
    sparql = re.sub(r'order by desc\s+....?_open\s+([\S]+)\s+....?_close', '_obd_ \\1', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'order by asc\s+....?_open\s+([\S]+)\s+....?_close', '_oba_ \\1', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'order by\s+([\S]+)', '_oba_ \\1', sparql, flags=re.IGNORECASE)
    return sparql


def reverse_shorten_query(sparql):
    sparql = re.sub(r'_oba_ ([\S]+)', 'order by asc (\\1)', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'_obd_ ([\S]+)', 'order by desc (\\1)', sparql, flags=re.IGNORECASE)
    return sparql


def read_template_file(file):
    annotations = list()
    line_number = 1
    with open(file) as f:
        for line in f:
            values = line[:-1].split(';')
            target_classes = [values[0] or None, values[1] or None, values[2] or None]
            question = values[3]
            query = values[4]
            generator_query = values[5]
            id = values[6] if (len(values) >= 7 and values[6]) else line_number
            line_number += 1
            annotation = Annotation(question, query, generator_query, id, target_classes)
            annotations.append(annotation)
    return annotations


class Annotation:
    def __init__(self, question, query, generator_query, id=None, target_classes=None):
        self.question = question
        self.query = query
        self.generator_query = generator_query
        self.id = id
        self.target_classes = target_classes if target_classes != None else []
        self.variables = extract_variables(generator_query)


def extract_variables(query):
    variables = []
    query_form_pattern = r'^.*?where'
    query_form_match = re.search(query_form_pattern, query, re.IGNORECASE)
    if query_form_match:
        letter_pattern = r'\?(\w)'
        variables = re.findall(letter_pattern, query_form_match.group(0))
    return variables


def extract_encoded_entities(encoded_sparql):
    sparql = decode(encoded_sparql)
    entities = extract_entities(sparql)
    encoded_entities = list(map(encode, entities))
    return encoded_entities


def extract_entities(sparql):
    triples = extractTriples(sparql)
    entities = set()
    for triple in triples:
        possible_entities = [triple['subject'], triple['object']]
        sorted_out = [e for e in possible_entities if not e.startswith('?') and ':' in e]
        entities = entities.union([re.sub(r'^optional{', '', e, flags=re.IGNORECASE) for e in sorted_out])
    return entities


def extract_predicates(sparql):
    triples = extractTriples(sparql)
    predicates = set()
    for triple in triples:
        pred = triple['predicate']
        predicates.add(pred)
    return predicates


def extractTriples(sparqlQuery):
    triples = []
    whereStatementPattern = r'where\s*?{(.*?)}'
    whereStatementMatch = re.search(whereStatementPattern, sparqlQuery, re.IGNORECASE)
    if whereStatementMatch:
        whereStatement = whereStatementMatch.group(1)
        triples = splitIntoTriples(whereStatement)
    return triples


def splitIntoTriples(whereStatement):
    tripleAndSeparators = re.split('(\.[\s\?\<$])', whereStatement)
    trimmed = [str.strip() for str in tripleAndSeparators]

    def repair(list, element):
        if element not in ['.', '.?', '.<']:
            previousElement = list[-1]
            del list[-1]
            if previousElement in ['.', '.?', '.<']:
                cutoff = previousElement[1] if previousElement in ['.?', '.<'] else ''
                list.append(cutoff + element)
            else:
                list.append(previousElement + ' ' + element)
        else:
            list.append(element)

        return list

    tripleStatements = reduce(repair, trimmed, [''])
    triplesWithNones = list(map(splitIntoTripleParts, tripleStatements))
    triples = [triple for triple in triplesWithNones if triple != None]
    return triples


def splitIntoTripleParts(triple):
    statementPattern = r'(\S+)\s+(\S+)\s+(\S+)'
    statementPatternMatch = re.search(statementPattern, triple)

    if statementPatternMatch:
        return {
            'subject': statementPatternMatch.group(1),
            'predicate': statementPatternMatch.group(2),
            'object': statementPatternMatch.group(3)
        }
    else:
        return None
