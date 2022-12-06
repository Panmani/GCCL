# import dependencies
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('english')
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

stemmer = PorterStemmer()
def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))
import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
# import jieba

import os, json
from apted import APTED, helpers
# This is a Python implementation of the APTED algorithm ,
# the state-of-the-art solution for computing the tree edit distance
import numpy as np
# import pandas as pd # External dependency: pip install pandas
from tqdm import tqdm
# Import packages
# from urllib.request import urlopen
# from bs4 import BeautifulSoup
# import re
from download import read_wiki_doc
from sklearn.model_selection import train_test_split

from config import *
os.environ["CORENLP_HOME"] = "../stanford-corenlp-4.2.2"
from stanfordnlp.server import CoreNLPClient
client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'parse'],
        memory='4G',
        endpoint='http://localhost:9001',
        output_format="json",
        be_quiet=True)

import spacy
nlp = spacy.load("en_core_web_lg")

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

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

def load_word_dataset(meta_data_list, split):
    data_list = []
    for index, cur_pair in enumerate(tqdm(meta_data_list)):
        wiki_doc1 = read_wiki_doc(cur_pair["doc_pair"][0])
        wiki_doc2 = read_wiki_doc(cur_pair["doc_pair"][1])
        cur_pair["doc_pair"] = [wiki_doc1, wiki_doc2]
        data_list.append(cur_pair)
    return data_list
df = pd.DataFrame()
df['text'] = ''
df['num'] = ''
""" Test on WORD dataset """

with open("fixed_meta_split.json", "r") as meta_file:
    train_meta, val_meta, test_meta = json.load(meta_file)

# with open(os.path.join(RAW_DATA_DIR, META_DATA_NAME), "r") as meta_file:
#     meta_data_list = json.load(meta_file)
#
# train_meta, test_meta = train_test_split(meta_data_list,
#                                                 test_size=0.2,
#                                                 random_state=42)

# load_word_dataset(train_meta, "train")
train_set = load_word_dataset(train_meta, "train")
test_set = load_word_dataset(test_meta, "test")


# data_path = "../../DATA/WORD/WORD.csv"
# if not os.access(data_path, os.R_OK):
#         print("Failed to read the target file: {}".format(data_path))

# data = pd.read_csv(data_path, encoding="ISO-8859-1")

SIM_THRESHOLD = 0.08
TRUNCATE_AT = 300
# 100: Cur Acc: 0.47314911366006257; Cur positive percentage: 0.14937434827945778
# 200: Cur Acc: 0.4760166840458811; Cur positive percentage: 0.14937434827945778
# 300: Cur Acc: 0.47758081334723673; Cur positive percentage: 0.14937434827945778
correct_pred = 0
pos_label_count = 0
test_size = 0
for index, cur_pair in enumerate(tqdm(train_set)):
    wiki_doc1, wiki_doc2 = cur_pair["doc_pair"]
    df.loc[index]= [wiki_doc1,index]
    df.loc[index+len(train_set)]= [wiki_doc2,-index]
print(df)
# clean text and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['text'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2-t1)/60, "min")
# first get a list of all words
all_words = [word for item in list(df['tokenized']) for word in item]
# use nltk fdist to get a frequency distribution of all words
fdist = FreqDist(all_words)
print(len(fdist)) # number of unique words
# choose k and visually inspect the bottom 10 words of the top k
k = 50000
top_k_words = fdist.most_common(k)
print(top_k_words[-10:])
# choose k and visually inspect the bottom 10 words of the top k
k = 15000
top_k_words = fdist.most_common(k)
print(top_k_words[-10:])
# define a function only to keep words in the top k words
top_k_words,_ = zip(*fdist.most_common(k))
top_k_words = set(top_k_words)
def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]
df['tokenized'] = df['tokenized'].apply(keep_top_k_words)
# document length
df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
doc_lengths = list(df['doc_len'])
df.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:",len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))
# plot a histogram of document length
num_bins = 1000
fig, ax = plt.subplots(figsize=(12,6));
# the histogram of the data
n, bins, patches = ax.hist(doc_lengths, num_bins, density=1, stacked=True)
ax.set_xlabel('Document Length (tokens)', fontsize=15)
ax.set_ylabel('Normed Frequency', fontsize=15)
ax.grid()
ax.set_xticks(np.logspace(start=np.log10(50),stop=np.log10(2000),num=8, base=10.0))
plt.xlim(0,2000)
ax.plot([np.average(doc_lengths) for i in np.linspace(0.0,0.0035,100)], np.linspace(0.0,0.0035,100), '-',
        label='average doc length')
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()
# only keep articles with more than 30 tokens, otherwise too short
df = df[df['tokenized'].map(len) >= 40]
# make sure all tokenized items are lists
df = df[df['tokenized'].map(type) == list]
df.reset_index(drop=True,inplace=True)
print("After cleaning and excluding short aticles, the dataframe now has:", len(df), "articles")
# create a mask of binary values
msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
train_df.reset_index(drop=True,inplace=True)

test_df = df[~msk]
test_df.reset_index(drop=True,inplace=True)
print(len(df),len(train_df),len(test_df))
def train_lda(data):
    """
    This function trains the lda model
    We setup parameters like number of topics, the chunksize to use in Hoffman method
    We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    num_topics = 100
    chunksize = 300
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    t2 = time.time()
    print("Time to train LDA model on ", len(df), "articles: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda
dictionary,corpus,lda = train_lda(train_df)
# show_topics method shows the the top num_words contributing to num_topics number of random topics
lda.show_topics(num_topics=10, num_words=20)
print(df)
print(test_df)
# select and article at random from test_df
random_article_index = np.random.randint(len(test_df))
print(random_article_index)
new_bow = dictionary.doc2bow(test_df.iloc[random_article_index,2])
print(test_df.iloc[random_article_index,1])
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])
# we need to use nested list comprehension here
# this may take 1-2 minutes...
doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
print(doc_topic_dist.shape)
def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))
def get_most_similar_documents(query,matrix,k=1):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims # the top k positional index of the smallest Jensen Shannon distances
# this is surprisingly fast
#most_sim_ids = get_most_similar_documents(new_doc_distribution,doc_topic_dist)
#most_similar_df = train_df[train_df.index.isin(most_sim_ids)]
#print(most_similar_df['index'].item())
TP = 0
TN = 0
FP = 0
FN = 0
Precision = 0
Recall = 0
Specificity = 0
F1 = 0
for random_article_index in range(len(test_df)):
    try:
        new_bow = dictionary.doc2bow(test_df.iloc[random_article_index,2])
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])
        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
        most_sim_ids = get_most_similar_documents(new_doc_distribution,doc_topic_dist)
        score = most_sim_ids[df[df.num == -test_df.iloc[random_article_index,1]].index.item()]
        if score < 0.7 and train_set[test_df.iloc[random_article_index,1]]['score'] > 0:
            TP = TP + 1
        elif score >= 0.7 and train_set[test_df.iloc[random_article_index,1]]['score'] <= 0:
            TN = TN + 1
        elif score < 0.7 and train_set[test_df.iloc[random_article_index,1]]['score'] <= 0:
            FN = FN + 1
        elif score >= 0.7 and train_set[test_df.iloc[random_article_index,1]]['score'] > 0:
            FP = FP + 1
        if TP + FP != 0:
            Precision = TP / (TP + FP)
        if TP + FN != 0:
            Recall = TP / (TP + FN)
        if TN + FP > 0:
            Specificity = TN / (TN+FP)
        if Precision + Recall != 0:
            F1 = 2 * Precision * Recall / (Precision + Recall)
        print("Index: {} --- Cur Acc: {}; Cur Precision: {}; Cur Recall: {}; Cur Specificity: {}; Cur F1-score: {};".format(random_article_index, (TP + TN) / (TP + TN + FP + FN),  Precision, Recall, Specificity, F1))
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(e)
        continue
