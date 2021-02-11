# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:15:13 2019

@author: mpfou
"""
import pandas as pd
from preprocess_data import DataPreprocess
import csv
import fasttext
from collections import Counter

# Importing libraries for WordCloud visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt



class ClassifyText:
    def __init__(self, train_data, class_mappings, dataframe, lr=0.01, dim=20, epoch=500,  word_ngrams=2, loss='hs'):
        
        #Training data file name
        self.train_data = train_data
        self.preprocess_train_data = self.train_data_preprocess()
        
        #Class mappings file name
        self.class_mappings=class_mappings
        self.table_class_mappings=self.to_dataframe(self.class_mappings)
        
        #Interview dataframe
        self.dataframe = dataframe
        
        #Learning Rate
        self.lr = lr

        # Dimension
        self.dim = dim

        # Epochs to train
        self.epoch = epoch

        # Word Ngrams
        self.word_ngrams=word_ngrams

        # Loss function at the output layer
        self.loss=loss
        
        self.classified_data, self.codes=self.classify(self.dataframe)
        
    def to_dataframe(self, filename):
        #Initializing an empty dataframe
        df = pd.DataFrame(columns=['Data','Label'])
        
        #Read the training data
        file = open(filename, "r")
        for each in file:
            if(each.startswith('__label__')):
                label=each.partition(' ')[0]
                detail=each.partition(' ')[2].replace('\n','')
                df=df.append({'Data': detail, 'Label': label}, ignore_index=True) 
        return df
    
    #Function for preprocessing the training data
    #The training data is a text file 
    #with each line starting with the label prefixed with "__label__" and then the text corresponding to that label
    def train_data_preprocess(self):
         
        train_df= self.to_dataframe(self.train_data)
        #Getting the list of classes from the data
        self.classes=train_df['Label'].unique()
        
        #Preprocess the training data
        preprocess_obj=DataPreprocess()
        train_df['Data_clean']=train_df['Data'].apply(preprocess_obj.preprocess).apply(preprocess_obj.join) 
        
        #Shuffle the data
        FinalDf=train_df.sample(n=train_df.shape[0])
        preprocess_train_data_file='preprocess_train.txt'
        
        #Save the preprocessed train data into a file 'preprocess_train.txt'
        FinalDf[['Label','Data_clean']].to_csv(preprocess_train_data_file, header=None, index=None, sep=' ', mode='a',quoting=csv.QUOTE_NONE, escapechar=' ')
        return preprocess_train_data_file
    
    #Function for getting the classes from the provided training data
    def get_classes(self):
        return self.classes
    
    #Function for text classification
    def classify(self,data):
        #Train the data with FastText
        model = fasttext.train_supervised(self.preprocess_train_data, lr=self.lr, dim=self.dim, epoch=self.epoch,  word_ngrams=self.word_ngrams, loss=self.loss)
        
        #Preprocess the combined column
        preprocess_obj=DataPreprocess()
        data=data.fillna('')
        data['Combined']=data['Question']+' '+data['Answer']
        data['Data_clean']=data['Combined'].apply(preprocess_obj.preprocess).apply(preprocess_obj.join)
        
        #Taking only the records in which there is atleast 3 words after preprocessing
        testdata=data[data['Data_clean'].map(lambda d: len(d.split(' '))) > 2][['Question','Answer','Data_clean']]
        
        #Predict the label
        labels = model.predict(testdata['Data_clean'].tolist(),k=1)
        codes=[]
        for each in labels[0]:
            codes.append(each[0])
            
        #Assigning the predicted labels for the respective rows
        testdata['Labels']=codes
        
        #Assiging the similarity value of each predicted labels
        testdata['Probs']=labels[1]
        
        codes=dict()
        print("WordCloud Visualization from FastText\n")
        self.table_class_mappings.set_index('Label',inplace=True)
        for each in self.classes:
            each_df=testdata[testdata['Labels']==each]
            print(self.table_class_mappings.loc[each]['Data'])
            # Getting keywords of each labels
            text = " ".join(review for review in each_df['Data_clean'])
            
            # Create and generate a word cloud image:
            try:
                wordcloud = WordCloud(background_color="white").generate(text)
                keywords=Counter(text.split())
                codes[each]=[i for i in keywords if keywords[i] > 10]
            
                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                
            except:
                pass
        return testdata,codes
    
    def get_keywords(self):
        return self.codes,self.table_class_mappings
    def get_classified_data(self):
        return self.classified_data