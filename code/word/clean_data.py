import os
import numpy as np
import yaml

with open("code/configs/config.yaml") as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

LEN_TH = 7

doc_path = os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'], yaml_config['DATASET']['WORD']['WIKI_DOC_DIR'])
all_doc_files = [fn for fn in os.listdir(doc_path) if os.path.isfile(os.path.join(doc_path, fn)) and fn[-3:] == "txt"]

exclude_list = ["Special_administrative_regions_of_China.txt"]

def get_actual_doc_start(fn, doc, len_threshold = LEN_TH):
    if len(doc) > 0 and doc[0] == ' ':
        found_second_upper = False
        if doc[1].isupper():
            for idx, char in enumerate(doc[2:]):
                if char.isupper():
                    actual_start = idx + 2
                    noise = doc[:actual_start]
                    noise_words = noise.split(" ")
                    # if len(noise_words) > len_threshold:
                    #     if fn not in non_skip_list:
                    #         return actual_start
                    #     else:
                    #         return 0
                    # else:
                    return actual_start
            return 0
        else:
            return 0
    else:
        return 0

def contain_title_words(title, doc):
    title_words = title.lower().split("_")
    doc_words = doc.lower().split(" ")
    for tw in title_words:
        if tw not in doc_words:
            print(tw)
            return False
    return True

# def noise_has_special_words(noise):
#     special_words = ["History"]
#     noise_words = noise.lower().split(" ")
#     for

noisy_count = 0
list_of_noise = []
for fn in all_doc_files:
    with open(os.path.join(doc_path, fn), "r", encoding="utf8") as doc_file:
        doc = doc_file.read()
        actual_start = get_actual_doc_start(fn, doc)
        noise = doc[:actual_start]
        noise_words = noise.split(" ")
        # if len(noise_words) > LEN_TH:
        if actual_start > 0:
            # if len(noise_words) < LEN_TH and fn not in exclude_list:
            # # if actual_start > 0:
            #     print('--------------------')
            # else:
            #     print('++++++++++++++++++++ (keep as is)')
            print('--------------------')
            if contain_title_words(fn[:-4], doc):
                print(">>> need to clean")
            else:
                print('(keep as is)')

            print(fn)
            print(fn[:-4])
            print(doc[:100])
            # noise = doc[:actual_start]
            print(noise + " |||| " + doc[actual_start:100])

            if noise not in list_of_noise:
                list_of_noise.append(noise)
            noisy_count += 1

        # else:



print(list_of_noise, len(list_of_noise))
print("Noisy / Total:", noisy_count, len(all_doc_files))
max_noise_len = max(list_of_noise, key = len)
print("max_noise_len:", max_noise_len)

noise_len_list = []
for noise in list_of_noise:
    noise_words = noise.split(" ")
    noise_len_list.append(len(noise_words))

# noise_len = np.array(noise_len_list)

for pc in range(0, 101, 5):
    print(pc, "th percentile: ", np.percentile(noise_len_list, pc))
