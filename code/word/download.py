""" Download IBM WORD dataset """
from tqdm import tqdm
import os
import pandas as pd # External dependency: pip install pandas

# Import packages
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import json
import yaml


# Scrape wikipedia pages
def extract_text_from_wiki_url(wiki_url):
    # Specify url of the web page
    source = urlopen(wiki_url).read()

    # Make a soup
    soup = BeautifulSoup(source,'lxml')

    # Extract the plain text content from paragraphs
    paras = []
    for paragraph in soup.find_all('p'):
        paras.append(str(paragraph.text))

    # # Extract text from paragraph headers
    # heads = []
    # for head in soup.find_all('span', attrs={'mw-headline'}):
    #     heads.append(str(head.text))

    # # Interleave paragraphs & headers
    # text = [val for pair in zip(paras, heads) for val in pair]
    # text = ' '.join(text)

    text = ' '.join(paras)

    # Drop footnote superscripts in brackets
    text = re.sub(r"\[.*?\]+", '', text)

    # Replace '\n' (a new line) with '' and end the string at $1000.
    text = text.replace('\n', '')
    return text

# def write_wiki_doc(wiki_doc_names, doc_name, doc, wiki_doc_path):
#     if doc_name not in wiki_doc_names:
#         with open(os.path.join(wiki_doc_path, doc_name + ".txt"), "w") as doc_file:
#             doc_file.write(doc)
#     wiki_doc_names.append(doc_name)
#     return wiki_doc_names

def disambiguate_doc_name(doc_name):
    if doc_name == "MUD":
        return "MUD_game"
    else:
        return doc_name

if __name__ == '__main__':

    with open("code/configs/config.yaml") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)


    if not os.path.isdir(yaml_config['DATASET']['WORD']['RAW_DATA_DIR']):
        os.mkdir(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'])

    wiki_doc_path = os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'], yaml_config['DATASET']['WORD']['WIKI_DOC_DIR'])
    if not os.path.isdir(wiki_doc_path):
        os.mkdir(wiki_doc_path)

    if not os.access(yaml_config['DATASET']['WORD']['CSV_PATH'], os.R_OK):
        print("Failed to read the target file: {}".format(yaml_config['DATASET']['WORD']['CSV_PATH']))
    data_csv = pd.read_csv(yaml_config['DATASET']['WORD']['CSV_PATH'], encoding="ISO-8859-1")

    meta_data_list = []
    wiki_doc_urls = {}
    for index, row in data_csv.iterrows():
        doc_name1 = disambiguate_doc_name(row["concept1 URI"].split("/")[-1])
        doc_name2 = disambiguate_doc_name(row["concept2 URI"].split("/")[-1])

        wiki_doc_urls[doc_name1] = row["concept1 URI"]
        wiki_doc_urls[doc_name2] = row["concept2 URI"]
        # # If the wiki documents already exist, use the following lines instead of downloading again
        # if not os.path.isfile(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, doc_name1 + ".txt")) or \
        #         not os.path.isfile(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, doc_name2 + ".txt")):
        #     continue

        # --------------------- create metadata entry ---------------------
        cur_pair = {
            "id"            : index,
            "concept_pair"  : (row["concept 1"], row["concept 2"]),
            "doc_pair"      : (doc_name1, doc_name2),
            "score"         : row['score'],
            "split"         : row['Train/Test'],
        }

        meta_data_list.append(cur_pair)

    pblm_wiki_doc_names = {'error' : [],
                           'empty' : [],}
    for doc_name in tqdm(wiki_doc_urls):
        # --------------------- Scrape wikipedia ---------------------
        try:
            wiki_doc = extract_text_from_wiki_url(wiki_doc_urls[doc_name])
            # wiki_doc2 = extract_text_from_wiki_url(row["concept2 URI"])
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            pblm_wiki_doc_names['error'].append(doc_name)
            print(doc_name, wiki_doc_urls[doc_name])
            print(e)
            continue

        if len(wiki_doc) == 0:
            pblm_wiki_doc_names['empty'].append(doc_name)
            print("Warning! empty entry:", doc_name)

        with open(os.path.join(wiki_doc_path, doc_name + ".txt"), "w") as doc_file:
            doc_file.write(wiki_doc)

        # wiki_doc_names = write_wiki_doc(wiki_doc_names, doc_name1, wiki_doc1, wiki_doc_path)
        # wiki_doc_names = write_wiki_doc(wiki_doc_names, doc_name2, wiki_doc2, wiki_doc_path)
    print("Problematic docs:")
    print(pblm_wiki_doc_names)

    clean_meta_data_list = []
    for meta in meta_data_list:
        if meta["doc_pair"][0] not in pblm_wiki_doc_names['error'] + pblm_wiki_doc_names['empty'] and \
            meta["doc_pair"][1] not in pblm_wiki_doc_names['error'] + pblm_wiki_doc_names['empty']:
            clean_meta_data_list.append(meta)
        else:
            print("Remove meta:", meta)

    with open(os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'], yaml_config['DATASET']['WORD']['META_DATA_NAME']), "w") as meta_file:
        json.dump(clean_meta_data_list, meta_file)
