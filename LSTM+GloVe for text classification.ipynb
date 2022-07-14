import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import torchtext
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

#loading the data
reviews = pd.read_csv("reviews.csv")#this is the imported csv file containing the data which is used for text classification
print(reviews.shape)#prints the dimensions(no of rows and columns)
reviews.head()#prints the first 5 rows

reviews['Title'] = reviews['Title'].fillna('')#fill all the Na with ''
reviews['Review Text'] = reviews['Review Text'].fillna('')#fill all the Na with ''
reviews['review'] = reviews['Title'] + ' ' + reviews['Review Text']#creating new review with title+' '+text

#keeping only relevant columns and calculating sentence lengths
reviews = reviews[['review', 'Rating']]
reviews.columns = ['review', 'rating']
reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
reviews.head()

#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}#changes the rating from 1 to 0 correspondingly 5 to 4f
reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])

#mean sentence length
np.mean(reviews['review_length'])

#doing tokenization and splitting the text and doing regex operations fr easier manipulation of data
#tokenization
tok = spacy.load("en_core_web_sm")
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

#count number of occurences of each word
counts = Counter()
for index, row in reviews.iterrows():
    counts.update(tokenize(row['review']))

#deleting infrequent words
print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))

#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
reviews.head()

#check how balanced the dataset is
Counter(reviews['rating'])

X = list(reviews['encoded'])
y = list(reviews['rating'])
from sklearn.model_selection import train_test_split#splitting the data sets into 2 train and 2 validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

train_ds = ReviewsDataset(X_train, y_train)#trainign dataset
valid_ds = ReviewsDataset(X_valid, y_valid)#validation dataset
batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)#dataloading into the varaible
val_dl = DataLoader(valid_ds, batch_size=batch_size)

def load_glove_vectors(glove_file="glove.6B.50d.txt"):#loading the glove.6b.50d vectors and converting them into forms usable by the code

    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            #print(split)
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors

def get_emb_matrix(pretrained, word_counts, emb_size = 50):#creating embeddings for the word vectors(here glove vector)
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx

word_vecs = load_glove_vectors()#define a variable and loading glove vector onto it
pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)#getting the values from embeddings for the glove vectors

class LSTM_glove_vecs(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

def train_model(model, epochs=10, lr=0.001): #training the model and validating it
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

def train_model_regr(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.float()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.mse_loss(y_pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss = validation_metrics_regr(model, val_dl)
        if i % 5 == 1:
            print("train mse %.3f val rmse %.3f" % (sum_loss/total, val_loss))

def validation_metrics_regr (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.float()
        y_hat = model(x, l)
        loss = np.sqrt(F.mse_loss(y_hat, y.unsqueeze(-1)).item())
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total

model = LSTM_glove_vecs(vocab_size, 50, 50, pretrained_weights)
train_model(model, epochs=30, lr=0.1)
train_model(model, epochs=30, lr=0.05)
train_model(model, epochs=30, lr=0.05)
