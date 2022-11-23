import os
import numpy as np
import yaml

with open("code/configs/config.yaml") as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

LEN_TH = 7

doc_path = os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'], yaml_config['DATASET']['WORD']['WIKI_DOC_DIR'])
all_doc_files = [fn for fn in os.listdir(doc_path) if os.path.isfile(os.path.join(doc_path, fn)) and fn[-3:] == "txt"]

noisy_count = 0
list_of_noise = []
for fn in all_doc_files:
    with open(os.path.join(doc_path, fn), "r", encoding="utf8") as doc_file:
        doc = doc_file.read()
        if len(doc) == 0:
            print(fn)
