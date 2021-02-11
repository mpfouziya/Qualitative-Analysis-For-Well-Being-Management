# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:19:56 2019

@author: mpfou
"""
import pandas as pd
from preprocess_data import DataPreprocess
from text_classify import ClassifyText
from convert_vector import Vectorization
import gensim.models.word2vec as w2v
import graphviz
from graphviz import render
import os

#We need the codes from the FastText, and it need to be given as input to Word2Vec to get the subcodes
#If pre_trained=False, then the newly created vectors will be saved in the specified file.
#If pre_trained=True, then the trained vectors will be retrieved from the specified file.
class TextualGraph:
    def __init__(self, train_data, class_mappings, data, filename, pre_trained=False, vector_dim=300, vector_epoch=1, min_word_count=3, context_size=7, downsampling=1e-3,
                 fast_text_epoch=500, fast_text_lr=0.01, fast_text_ngrams=2, fast_text_loss='hs' ):
        
        self.vector_dim=vector_dim
        self.vector_epoch=vector_epoch
        self.min_word_count=min_word_count
        self.context_size=context_size
        self.downsampling=downsampling
        self.fast_text_epoch=fast_text_epoch
        self.fast_text_lr=fast_text_lr
        self.fast_text_ngrams=fast_text_ngrams
        self.fast_text_loss=fast_text_loss
        
        self.train_data=train_data
        self.class_mappings=class_mappings
        self.data=pd.read_csv(data, encoding='utf-8')
        self.pre_trained=pre_trained
        self.filename=filename
        
        self.start_model()
        
    #Function for getting the vectors and codes
    def start_model(self):      
        
        #Classify the provided data
        classify=ClassifyText(train_data=self.train_data, class_mappings=self.class_mappings, dataframe=self.data, lr=self.fast_text_lr, dim=self.vector_dim, 
                              epoch=self.fast_text_epoch, word_ngrams=self.fast_text_ngrams, loss=self.fast_text_loss)
        self.codes, self.table_class_mappings=classify.get_keywords()
        self.classified_data=classify.get_classified_data()
        
        #Preprocess the answer columns
        preprocess_obj=DataPreprocess()
        self.classified_data['Answer_clean']=self.classified_data['Answer'].apply(preprocess_obj.preprocess).apply(preprocess_obj.join)
        
        #Do or get vectors
        if(self.pre_trained):
            #Function for loading the saved vector model
            self.response2vec = w2v.Word2Vec.load(self.filename)
        else:
            vect=Vectorization(self.data, filename=self.filename, epochs=self.vector_epoch, num_features=self.vector_dim, 
                               min_word_count=self.min_word_count, context_size=self.context_size, downsampling=self.downsampling)
            self.response2vec=vect.get_vectors()
        self.visualize_graph()
        
    #Function for cleaning the branch to remove duplicate elements
    def clean_branch(self, keyword, branch):
        for b in branch:
            aa=' '
            if(b[0].endswith('ing') | b[0].endswith('ed') | b[0].endswith('ry') | b[0].endswith('s')):
                if(b[0].endswith('ing')):
                    aa=b[0][:-3]
                else:
                    aa=b[0][:-2]
            for i in branch:
                if(((i[0].startswith(aa)) & (i[0]!=b[0]))):
                    branch.remove(i)
        
                #Removing the duplicate codes from codes and subcodes
                if((keyword.startswith(aa)) | (b[0].startswith(keyword)) | (keyword.startswith(b[0])) | (b[0].endswith('ben'))): 
                    if(b in branch):
                        branch.remove(b)
        return branch


    #Fucntion to clean the edges formed when there is duplication
    def clean_edges(self, edges):
        for each in edges:
        #Split each element to check whether the list contain the same element in swapped form 
            splitted=each.split(' ')
            if(len(splitted)>1):
                swap=splitted[1]+' '+splitted[0]
                if(swap in edges):
                    edges.remove(swap)         
        return edges 


    #Function for plotting the codes and subcodes  
    def plot_graph(self, code, df, keyword):
        edges=[]
        #Initializing the graph
        code_graph = graphviz.Digraph(strict=True)
        root=code
  
        #Creating the root and next level node with the code
        code_graph.node(root,color='red')
        code_graph.node(keyword,color='blue')
  
        #Obtaining the branch from the root node
        branch=self.response2vec.most_similar(keyword)
        branch=self.clean_branch(keyword,branch)
  
        #Finding the codes in the branch are related with the respective category
        for b in branch: 
            subcodes=[]
            subcodes1=[]
            temp=self.classified_data[(self.classified_data['Data_clean'].str.contains(keyword)) & (self.classified_data['Data_clean'].str.contains(b[0]))]  
            details=[]
            for ans in temp['Answer']:
                details.append(ans)
            if(len(details)>0):  
                code_graph.edge(root,keyword)
                branch1=self.response2vec.most_similar(b[0])
      
      
                #Drilling down to find subcodes from codes
                for b1 in branch1:
                    temp1=self.classified_data[(self.classified_data['Answer_clean'].str.contains(b[0]+' '+b1[0])) | (df['Answer_clean'].str.contains(b1[0]+' '+b[0]))]  
                    details1=[]
                    for ans in temp1['Answer']:
                        details1.append(ans)
          
        
                    #Checking whether the codes and subcodes are present in the respective category
                    if(len(details1)>0):
                        for b11 in branch1:
                            temp2=self.classified_data[(self.classified_data['Answer_clean'].str.contains(b[0]+' '+b1[0]+' '+b11[0]))]
                            details2=[]
                            for ans in temp2['Answer']:
                                details2.append(ans)
                            if(len(details2)>0):
                                subcodes.append(b[0]+' '+b1[0]+' '+b11[0])
                                extra=b[0]+' '+b1[0]
                                subcodes=list(filter(lambda a: a != extra, subcodes))
                                break
                            else:
                                if(b[1]>0.55):
                                    subcodes.append(b[0]+' '+b1[0])
                        break
                    else:
                        if((b[0] not in subcodes1) & (b[1]>0.55)):
                            subcodes1.append(b[0])
            
      
                #Creating an edge from codes to subcodes in the graph.
                if(len(subcodes)>0):
                    edges.append(subcodes[0])
                elif(len(subcodes1)>0):
                    edges.append(subcodes1[0])
        
        #Cleaning the edge      
        clean_edge=self.clean_edges(edges)  
        for each in clean_edge:
            code_graph.edge(keyword,each)
    
        if(len(clean_edge)>0):
            #render the graph
            code_graph.render(os.path.join('VisualGraphs/'+code, keyword+'.gv'), format='png',view=True)
        
    
    def visualize_graph(self):
        for each in self.codes:
            key_code=self.table_class_mappings.loc[each]['Data']
            os.makedirs(os.path.join('VisualGraphs', key_code))
            each_df=self.classified_data[self.classified_data['Labels']==each]
            for subkey in self.codes[each]:
                self.plot_graph(key_code, each_df, subkey)
