import json 

class Data: 
    def __init__(self, path): 
        self.path = path 
    
    def getList(self): 
        data = []
        with open(self.path, encoding="utf8") as file: 
            for line in file: 
                data.append(line.rstrip().lower()) 
        return data 

class Vocab: 
    def __init__(self, data): 
        self.data = data 
    
    def getVocab(self): 
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'unknown': 3}
        index = 4 
        for line in self.data: 
            words = line.split()
            for word in words: 
                if word in vocab.keys():
                    continue
                else: 
                    vocab[word] = index 
                    index = index + 1
        
        return vocab 
    

train_en = r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\Data\data\train.en'
train_vi = r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\Data\data\train.vi'
test_en = r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\Data\data\tst2013.en'
test_vi = r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\Data\data\tst2013.vi'

train_tool_en = Data(train_en) 
train_tool_vi = Data(train_vi)
test_tool_en = Data(test_en)
test_tool_vi = Data(test_vi)   

en_sen = train_tool_en.getList() + test_tool_en.getList()
vi_sen = train_tool_vi.getList() + test_tool_vi.getList()

en = Vocab(en_sen) 
vi = Vocab(vi_sen)

en_vocab = en.getVocab()
vi_vocab = vi.getVocab()


with open(r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\Data\data\SRC.json', 'w') as fp:
    json.dump(en_vocab, fp)

with open(r'C:\Users\Admin\DEEP_LEARNING\Project\Transformer\Data\data\TGT.json', 'w') as fp:
    json.dump(vi_vocab, fp)



