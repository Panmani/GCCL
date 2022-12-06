""" Download IBM WORD dataset """
from tqdm import tqdm
import os
import pandas as pd # External dependency: pip install pandas

# Import packages
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import json

from config import *


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

    # Extract text from paragraph headers
    heads = []
    for head in soup.find_all('span', attrs={'mw-headline'}):
        heads.append(str(head.text))

    # Interleave paragraphs & headers
    text = [val for pair in zip(paras, heads) for val in pair]
    text = ' '.join(text)


    # Drop footnote superscripts in brackets
    text = re.sub(r"\[.*?\]+", '', text)

    # Replace '\n' (a new line) with '' and end the string at $1000.
    text = text.replace('\n', '')
    return text

def write_wiki_doc(wiki_doc_names, doc_name, doc):
    if doc_name not in wiki_doc_names:
        with open(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, doc_name + ".txt"), "w") as doc_file:
            doc_file.write(doc)
    wiki_doc_names.append(doc_name)
    return wiki_doc_names


if __name__ == '__main__':

    if not os.path.isdir(RAW_DATA_DIR):
        os.mkdir(RAW_DATA_DIR)

    wiki_doc_path = os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR)
    if not os.path.isdir(wiki_doc_path):
        os.mkdir(wiki_doc_path)

    if not os.access(CSV_PATH, os.R_OK):
        print("Failed to read the target file: {}".format(CSV_PATH))
    data_csv = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

    meta_data_list = []
    wiki_doc_names = []
    for index, row in tqdm(data_csv.iterrows(), total=data_csv.shape[0]):
        doc_name1 = row["concept1 URI"].split("/")[-1]
        doc_name2 = row["concept2 URI"].split("/")[-1]

        # --------------------- Scrape wikipedia ---------------------
        try:
            wiki_doc1 = extract_text_from_wiki_url(row["concept1 URI"])
            wiki_doc2 = extract_text_from_wiki_url(row["concept2 URI"])
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            continue

        wiki_doc_names = write_wiki_doc(wiki_doc_names, doc_name1, wiki_doc1)
        wiki_doc_names = write_wiki_doc(wiki_doc_names, doc_name2, wiki_doc2)

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

    with open(os.path.join(RAW_DATA_DIR, META_DATA_NAME), "w") as meta_file:
        json.dump(meta_data_list, meta_file)
