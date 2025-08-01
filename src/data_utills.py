from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import pandas
import uuid
import sys


def label_to_num(label):
    if label == "negative":
        return 0
    if label == "positive":
        return 1
    if label =="neutral":
        return 2
    if label == "conflict":
       return 2

def process_ori_sentence(sentence,term,k=1):
    if not isinstance(term, str):
        term = str(term)

    pre_text = sentence[:sentence.index(term)]
    post_text = sentence[sentence.index(term)+len(term):]
    pre_text = pre_text.split()[::-1][:k][::-1]
    post_text =  post_text.split()[:k]
    text = sentence.replace(" ".join(pre_text),"").replace(" ".join(post_text),"").replace(term,"")

    return text

def process_arts_sentence(sentence,pre_index,post_index,k=1):
    pre_text = sentence[:pre_index]
    post_text = sentence[post_index:]
    pre_text = pre_text.split()[::-1][:k][::-1]
    post_text =  post_text.split()[:k]
    text = sentence.replace(" ".join(pre_text),"").replace(" ".join(post_text),"").replace(sentence[pre_index:post_index],"")
    return text


def read_json_ori(file_folder):
    data = {}
    for file in os.listdir(file_folder):
        with open(file_folder+"/"+file,"r",encoding="utf-8") as f:
            dataset = pandas.DataFrame()
            sentences = []
            aspects = []
            polarities = []
            ids = []
            sentences_x0 = []
            datasets = json.loads(f.read())
            for i in datasets:
                polarities.append(datasets[i]["polarity"])
                sentence = datasets[i]["sentence"]
                term = datasets[i]["term"]
                sentences_x0.append(process_ori_sentence(sentence,term))
                aspects.append(datasets[i]["term"])
                sentences.append(datasets[i]["sentence"])
                ids.append(datasets[i]["id"])
            dataset["sentences"] = sentences
            dataset["sentences_x0"] = sentences_x0
            dataset["aspects"] = aspects
            dataset["polarities"] = polarities
            dataset["ids"] = ids
            data[file[:-5]] = dataset
    return data



def read_csv_ori(file_folder):
    data = {}
    data_test = {} 

    for file in os.listdir(file_folder):
        if not file.endswith(".csv"):
            continue
        if file == "test":
            continue
        dataset = pandas.read_csv(os.path.join(file_folder, file))
        dataset['sentences_x0'] = [process_ori_sentence(sentence, term) for sentence, term in zip(dataset['sentences'], dataset['aspects'])]
        dataset['ids'] = [uuid.uuid4().hex for _ in range(len(dataset))]

        data[file[:-4]] = dataset

    for file_test in os.listdir(os.path.join(file_folder, "test")):
        dataset_test = pandas.read_csv(os.path.join(file_folder, "test", file_test))
        dataset_test['sentences_x0'] = [process_ori_sentence(sentence, term) for sentence, term in zip(dataset_test['sentences'], dataset_test['aspects'])]
        dataset_test['ids'] = [uuid.uuid4().hex for _ in range(len(dataset_test))]

        data_test[file_test[:-4]] = dataset_test

    return data, data_test

def  read_csv_arts(file_folder):
    data = {}
    for file in os.listdir(file_folder):
        lang = file[:-4]
        dataset = pandas.read_csv(os.path.join(file_folder, file))
        dataset['sentences_x0'] = [process_ori_sentence(sentence, term) for sentence, term in zip(dataset["sentences"], dataset['aspects'])]
        dataset['ids'] = [uuid.uuid4().hex for _ in range(len(dataset))]

        data[lang] = dataset

    return data

def read_json_arts(file):
    data = {}
    with open(file+"_test_enriched.json","r",encoding="utf-8") as f:
        dataset = pandas.DataFrame()
        sentences = []
        aspects = []
        polarities = []
        ids = []
        sentences_x0 = []
        datasets = json.loads(f.read())
        for i in datasets:
            polarities.append(datasets[i]["polarity"])
            aspects.append(datasets[i]["term"])
            sentence = datasets[i]["sentence"]
            pre_index = datasets[i]["from"]
            post_index = datasets[i]["to"]
            sentences_x0.append(process_arts_sentence(sentence,pre_index,post_index))
            sentences.append(datasets[i]["sentence"])
            ids.append(i)
        dataset["sentences"] = sentences
        dataset["sentences_x0"] = sentences_x0
        dataset["aspects"] = aspects
        dataset["polarities"] = polarities
        dataset["ids"] = ids
        data["test"] = dataset
    return data

class ABSADataset(Dataset):
    def __init__(self, sentences, sentences_x0,aspects, polarities, ids,tokenizer, max_len,max_len_a):
        self.sentences = sentences
        self.sentences_x0 = sentences_x0
        self.aspects = aspects
        self.polarities = polarities
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_len_a = max_len_a
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        sentence_x0 = str(self.sentences_x0[item])
        aspect = str(self.aspects[item])
        polarity = self.polarities[item]
        id = str(self.ids[item])
        encoding_all = self.tokenizer.encode_plus(
            sentence,aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = "max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        encoding_text = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = "max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        encoding_aspect = self.tokenizer.encode_plus(
            aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = "max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'sentence': sentence,
            'aspect':aspect,
            'id':id,
            'all_input_ids': encoding_all['input_ids'].flatten(),
            'all_attention_mask': encoding_all['attention_mask'].flatten(),
            'text_input_ids': encoding_text['input_ids'].flatten(),
            'text_attention_mask': encoding_text['attention_mask'].flatten(),
            'aspect_input_ids': encoding_aspect['input_ids'].flatten(),
            'aspect_attention_mask': encoding_aspect['attention_mask'].flatten(),
            'polarities': torch.tensor(label_to_num(polarity), dtype=torch.long)
     }
    
def create_data_loader(df, tokenizer, max_len,max_len_a, batch_size):
    ds = ABSADataset(
        sentences=df.sentences.to_numpy(),
        sentences_x0=df.sentences_x0.to_numpy(),
        aspects=df.aspects.to_numpy(),
        polarities=df.polarities.to_numpy(),
        ids=df.ids.to_numpy(),
        tokenizer=tokenizer,
        max_len= max_len,
        max_len_a = max_len_a
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

def load_data(dataset_name, type):
    afriData = ["afri_laptop", "afri_rest"]
    
    if type == "ORI":
#        data = read_json_ori("../dataset/SemEval2014/"+dataset_name)
        if dataset_name in afriData:

            data, test = read_csv_ori("../dataset/SemEval2014/afriData/"+dataset_name)
        else:
            data, test = read_csv_ori("../dataset/SemEval2014/"+dataset_name)
        train = data["train"]
        dev = data["dev"]
        test = test

        return train, test, dev
    if type == "ARTS":
        adv1_data = read_csv_arts(f"../dataset/ARTS_data/afriData/{os.path.join(dataset_name, 'adv1')}")
        adv2_data = read_csv_arts(f"../dataset/ARTS_data/afriData/{os.path.join(dataset_name, 'adv2')}")
        adv3_data = read_csv_arts(f"../dataset/ARTS_data/afriData/{os.path.join(dataset_name, 'adv3')}")

        return adv1_data, adv2_data, adv3_data
