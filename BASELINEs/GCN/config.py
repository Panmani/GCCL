# ---------- WORD dataset file ----------
CSV_PATH = ""

# ---------- Download destination / Cache ----------
# Using download.py, WORD dataset is downloaded to:
# RAW_DATA_DIR = "./WORD_raw/"
RAW_DATA_DIR = ""
WIKI_DOC_DIR = "wiki_docs"
META_DATA_NAME = "meta_data.json"

# ---------- DATA ----------
FIXED_SPLIT_FILE = "fixed_meta_split.json"
SCORE_THRESHOLD = 0.0

# ---------- BERT ----------
MODEL_NAME = 'bert-base-uncased'
DEFAULT_CKPT_DIR = "ckpt"
HISTORY_NAME = 'history.json'
# MAX_SEQ_LEN is used by the bert model to allocate the [SEP] token,
# not to be confused with the max length of docs from the dataset, args.max_len
# args.max_len <= MAX_SEQ_LEN
MAX_SEQ_LEN = 32
MAX_SENT_NUM = 20
NUM_EPOCHS = 5
BATCH_SIZE = 2
