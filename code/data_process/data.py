import pandas as pd 
import numpy as np 
import re 
import json 

class DataProcessing: 
    def __init__(self, path): 
        self.path = path

    def read_data(self):
        data = []
        with open(self.path, 'r', encoding='utf-8') as file:
            for line in file: 
                data.append(line.strip())
        return data 

    def process_sentence(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r',', '', sentence)
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = re.sub(r"\.$", "", sentence)
        sentence = re.sub(r"\!$", "", sentence)
        sentence = re.sub(r"\?$", "", sentence)
        sentence = sentence.lower()
        
        return sentence

    def make_dict(self): 
        data = self.read_data()

        word_to_num = {'<PAD>': 0, '<START>': 1, '<EOS>': 2, '<UNK>': 3} 
        num_to_word ={int(0): '<PAD>', int(1): '<START>', int(2): '<EOS>', int(3): '<UNK>'}

        
        for line in data: 
            line = self.process_sentence(line)
            words = line.split()
            for word in words: 
                if word not in word_to_num: 
                    num_to_word[int(len(word_to_num))] = word
                    word_to_num[word] = len(word_to_num)
        
        return word_to_num, num_to_word
                    
en = DataProcessing(path=r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\data\en_sents.txt')
vi = DataProcessing(path=r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\data\vi_sents.txt')

word_num_en, num_word_en  = en.make_dict()
word_num_vi, num_word_vi  = vi.make_dict()

with open(r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\data\word_num_en.json', 'w') as file:
    json.dump(word_num_en, file)
with open(r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\data\num_word_en.json', 'w') as file:
    json.dump(num_word_en, file)
with open(r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\data\num_word_vi.json', 'w') as file:
    json.dump(num_word_vi, file)     
with open(r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\data\word_num_vi.json', 'w') as file:
    json.dump(word_num_vi, file) 

print(len(num_word_en), len(word_num_en))
print(len(num_word_vi), len(word_num_vi))

