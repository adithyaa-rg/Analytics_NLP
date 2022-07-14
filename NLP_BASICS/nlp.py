!pip install unidecode
!pip install word2number
!pip install contractions
!pip install nlp
!pip install spacy
!pip install re

import nltk
import pandas
import re
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n #converet word to numeric
import contractions
nlp=nlp = spacy.load("en_core_web_sm")
from nltk.stem import PorterStemmer

text="Hi this is V. Look at what's happening here"
uc=re.findall("[A-Z]\w+",text)
print(uc)
#similarly re.split can split based on any required set of letters/points
#re.sub can replace any letter or set of letters by corresponding new letters
#re.search finds the firsst match of a particular word and returns the operation accoidingly
print(word_tokenize(text)) #this splits the words into individual parts
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")#the htmp parser function strips the htmp parts from texts
    stripped_text = soup.get_text(separator=" ")
    return stripped_text
text=strip_html_tags(text)
print(word_tokenize(text)) #this splits the words into individual parts

def expand_contractions(text):#expand stuff like i'm to i am
    text = contractions.fix(text)
    return text
text=expand_contractions(text)
print(word_tokenize(text)) #this splits the words into individual parts

doc = nlp(text)#lemmatization
mytokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc]# is is converted to be and similar stuff happens
print(mytokens) 

ps = PorterStemmer()
 
words = word_tokenize(text)
 
for w in words:
    print(w, " : ", ps.stem(w))
    
readcsv=pandas.read_csv("username.csv")
readcsv
readcsv=pandas.read_csv("username.csv",nrows=4)
readcsv
readcsv=pandas.read_csv("username.csv",na_values=["not available","n.a."])
#readcsv=pandas.read_csv("username.csv",na_values={"Username";["not available","n.a."],"Identifier";-1})
readcsv
readcsv.to_csv("newfile.csv")
