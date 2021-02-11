# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:23:31 2019

@author: mpfou
"""


from text_graph import TextualGraph

if __name__=='__main__':
    #The only statement that you need to call for obtaining the graphs
    #The mode is pre_trained=True
    #This will produce a folder 'VisualGraphs' with 7 category subfolders.
    #Within the subfolder we can see the graphs
    #'response2vec300D20.w2v' is the pretrained vector model 
    TextualGraph(train_data='train_data.txt',class_mappings='class_mappings.txt', data='CleanData.csv',filename='response2vec300D20.w2v',pre_trained=True)