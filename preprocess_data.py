# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:10:25 2019

@author: mpfou
"""
import nltk
import re

class DataPreprocess:
    def __init__(self):
        
        #Initializing the stop word list
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        #As the nltk stopwords list is not complete, add more stopwords to it.
        self.stop_words.add('yes')
        self.stop_words.add('yeah')
        self.stop_words.add('would')
        self.stop_words.add('could')
        self.stop_words.add('okay')
        self.stop_words.add('also')
        self.stop_words.add('ok')
        self.stop_words.add('oh')
        self.stop_words.add('th')
        self.stop_words.add('alright')
        self.stop_words.add('without')
        self.stop_words.add('might')
        self.stop_words.add('many')
        self.stop_words.add('much')
        self.stop_words.add('may')
        self.stop_words.add('per')
        self.stop_words.add('otherwise')
        
        #Initializing the lemmatizer
        self.wn=nltk.WordNetLemmatizer()
           
#Function for preprocessing
    def preprocess(self,text):
        #Removing the background information from the data
        text=re.sub("\\(.*?\\)","",text)
        
        #Tokenization
        word_tokens = nltk.word_tokenize(text) 
        
        #POS Tagging
        pos=nltk.pos_tag(word_tokens)
        j=0
        
        #Named Entity Recognition
        for i,k in zip(pos[0:], pos[1:]):
            if ((i[1]=='NNP') & (k[1]=='NNP')):
                word_tokens[j]=i[0]+'-'
            j=j+1   
        text=' '.join(word_tokens)
        text=re.sub('- ','-',text)
        text=re.sub("-","",text)
        
        #Removing non-alphabetic characters
        text = re.sub("[^a-zA-Z]"," ", text)
        word_tokens= nltk.word_tokenize(text)
        filtered_sentence = [] 
        for each in word_tokens:
            if(each.lower() not in self.stop_words):
                filtered_sentence.append(each)         
        text=[]    
        
        #Lemmatization
        for word in filtered_sentence:
            if(word.endswith('ss')==False):
                temp=self.wn.lemmatize(word.lower())
                if(len(temp)>1):
                    text.append(temp)
            else:
                text.append(word.lower())
    
        return text

    #Function for joining tokenized text
    def join(self,text):
        text=' '.join(text)
        return text                     