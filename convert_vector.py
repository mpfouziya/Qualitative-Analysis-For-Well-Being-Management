# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:11:39 2019

@author: mpfou
"""
import multiprocessing
from preprocess_data import DataPreprocess
import numpy as np
import gensim.models.word2vec as w2v
import sklearn.manifold
import pandas as pd
import seaborn as sns


class Vectorization:
    def __init__(self, dataframe, filename, epochs=1, num_features=300, min_word_count=3, context_size=7, downsampling=1e-3):
        
        #Initializing the dataframe
        self.dataframe = dataframe
        
        #Filename to where the vectors to be stored
        self.filename=filename
        
        #more dimensions = more generalized
        self.num_features = num_features
        
        # Minimum word count threshold.
        self.min_word_count = min_word_count

        # Number of threads to run in parallel.
        #more workers, then faster we can train
        self.num_workers = multiprocessing.cpu_count()

        # Context window length.
        self.context_size = context_size

        # Downsample setting for frequent words.
        self.downsampling = downsampling

        # Seed for the RNG, to make the results reproducible.
        self.seed = 1
        
        #Number of epochs to train
        self.epochs=epochs
        self.vectorize()
        
    #Format the column to vectorize into a list after preprocessing it
    def form_the_list(self):
        #Combine the qustion and the answer.
        self.dataframe=self.dataframe.fillna('')
        self.dataframe['Combined']=self.dataframe['Question']+' '+self.dataframe['Answer']
        
        #Preprocess the combined column
        preprocess_obj=DataPreprocess()
        preprocessed_column=self.dataframe['Combined'].apply(preprocess_obj.preprocess)
        self.data_to_vectorize=[]
        for each in preprocessed_column:
            if ((each is not np.nan) and (len(each)>0)):
                self.data_to_vectorize.append(each)
        #return data_to_vectorize
    
    #Function for vectorization
    def vectorize(self):
        self.response2vec = w2v.Word2Vec(
            sg=1,
            seed=self.seed,
            workers=self.num_workers,
            size=self.num_features,
            min_count=self.min_word_count,
            window=self.context_size,
            sample=self.downsampling
          )
        
        #Need to vectorize the contents of 'Combined' column which contains both Question and Answer
        self.form_the_list()
        self.response2vec.build_vocab(self.data_to_vectorize)
        self.response2vec.train(self.data_to_vectorize,total_examples=self.response2vec.corpus_count,epochs=self.epochs)
        
        #Save trained Word2Vec in a specified file.
        #So that we can retrieve and use it whenever we want

        self.response2vec.save(self.filename)
        self.visualize_tsne()
        
    #Function for returning the vectors    
    def get_vectors(self):
        return self.response2vec
        
    #Function for visualizing it using TSNE
    def visualize_tsne(self):
        tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
        all_word_vectors_matrix = self.response2vec.wv.vectors
        
        #Compress multidimensional matrix to 2-D
        all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
        
        points = pd.DataFrame(
            [
                (word, coords[0], coords[1])
                for word, coords in [
                    (word, all_word_vectors_matrix_2d[self.response2vec.wv.vocab[word].index])
                    for word in self.response2vec.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
        )
        
        sns.set_context("poster")
        print("TSNE plot of vectors")
        points.plot.scatter("x", "y", s=10, figsize=(20, 12))