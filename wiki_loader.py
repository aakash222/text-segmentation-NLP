from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer
from pathlib2 import Path
import re
import json
import os
import glob

with open('data_config.json') as json_file:
    config = json.load(json_file)
    dataset_path = config['data_path']
    
missing_stop_words = set(['of', 'a', 'and', 'to'])
section_delimiter = "========"
segment_seperator = "========"

def get_list_token():
    return "***LIST***"

def get_formula_token():
    return "***formula***"

def get_codesnipet_token():
    return "***codice***"

def get_special_tokens():
    special_tokens = []
    special_tokens.append(get_list_token())
    special_tokens.append(get_formula_token())
    special_tokens.append(get_codesnipet_token())
    return special_tokens

def get_seperator_foramt(levels = None):
    level_format = '\d' if levels == None else '['+ str(levels[0]) + '-' + str(levels[1]) + ']'
    seperator_fromat = segment_seperator + ',' + level_format + ",.*?\."
    return seperator_fromat


def extract_sentence_words(sentence, remove_missing_emb_words = False, remove_special_tokens = False):
    if (remove_special_tokens):
        for token in get_special_tokens():
            # Can't do on sentence words because tokenizer delete '***' of tokens.
            sentence = sentence.replace(token, "")
    tokenizer = RegexpTokenizer(r'\w+')
    sentence_words = tokenizer.tokenize(sentence)
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words



def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder+"/paths_cache.txt"
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    
    cache_file_path = get_cache_path(wiki_folder)

    with open(cache_file_path,'w') as f:
        for file in files:
            f.write(str(file) + u'\n')


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = get_seperator_foramt() if high_granularity else get_seperator_foramt(
        (1, 2))
    if not high_granularity:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)

        #delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')


    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]

    return non_empty_sections


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.strip()

    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]

    return sections

def read_wiki_file(path, n_context_sent = 1, remove_preface_segment=True, high_granularity=True):
    data = []
    targets = []
    all_sections = get_sections(path, high_granularity)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]

    list_sentence = get_list_token() + "."
    final_sentences = []
    label = []
    for section_ind in range(len(required_non_empty_sections)):
        sentences_ = required_non_empty_sections[section_ind].split('\n')
        sentences = [x for x in sentences_ if x != list_sentence]
        if sentences:
            for sentence in sentences[:-1]:
                final_sentences.append(sentence)
                label.append(0)
            final_sentences.append(sentences[-1])
            label.append(1)
    
    if len(final_sentences)>n_context_sent:
        for sent_ind in range(n_context_sent,len(final_sentences)):
            prev_context = final_sentences[sent_ind-n_context_sent:sent_ind]
            after_context = final_sentences[sent_ind: min(len(final_sentences),sent_ind+n_context_sent)]
            
            prev_context = " ".join(prev_context)
            after_context = " ".join(after_context)
            data.append([prev_context, after_context])
            targets.append(label[sent_ind-1])

    return data, targets, path


class WikipediaDataSet(Dataset):
    def __init__(self, root,n_context_sent = 1, train=True, high_granularity=False):

        root_path = root
        print(root_path)
        cache_path = get_cache_path(root_path)
        print(cache_path)
        if not os.path.exists(cache_path):
            print("loading names of all files")
            cache_wiki_filenames(root_path)
        self.textfiles = Path(cache_path).read_text().splitlines()

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.high_granularity = high_granularity
        self.n_context_sent = n_context_sent

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_wiki_file(Path(path),n_context_sent = self.n_context_sent,high_granularity=self.high_granularity)

    def __len__(self):
        return len(self.textfiles)
    