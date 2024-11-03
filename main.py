import torch
from flask import Flask, request, jsonify
from torch import nn
from gensim.models import KeyedVectors
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import nltk
import re


nltk.download('stopwords')
app = Flask(__name__)

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

word2vec = KeyedVectors.load("glove-twitter-100.kv")
word2idx = {word: idx for idx, word in enumerate(word2vec.index_to_key)}

max_len = 50

def encode(word):
    if word in word2idx.keys():
        return word2idx[word]
    return word2idx["unk"]

def lemmatize(doc):
    doc = re.sub(patterns, ' ', str(doc))
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token)
    
    return tokens

class ToxicRnn(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, num_classes):
        super(ToxicRnn, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, dropout=0.2, batch_first=True)
        
        self.cls = nn.Linear(hidden_size, num_classes)
        self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(word2vec.vectors), freeze=False)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.embedding_layer(x)

        output, hidden = self.lstm(x)
        output = self.batch_norm(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout(output)
        output = output[:, -1, :]
        output = self.cls(output)
        return output

@app.route('/')
def home():
    return 'Классификация токсичных комментариев'

@app.route('/predict', methods=['GET', 'POST'])
def predict():  
    if request.method == 'POST':
        X = request.json
        sentence = [encode(word) for word in lemmatize(X['sentence'])]
        t = torch.nn.functional.pad(torch.tensor(sentence), (0, max_len - len(sentence)))
        model = ToxicRnn(embed_size=100, hidden_size=64, num_layers=1, num_classes=2) 
        model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
        model.eval() 

        if torch.argmax(model(t.unsqueeze(0))) == 1:
            return jsonify({'predict' : 'toxic'})
        else:
            return jsonify({'predict' : 'no toxic'})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)