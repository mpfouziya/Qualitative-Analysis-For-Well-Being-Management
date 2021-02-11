# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import docx2txt
import re

class InterviewDataPreparation:
    def __init__(self,folderName):
        self.folderName=folderName
        
#Function for converting the data into dataframe
    def to_dataframe(self):
        alist=[]
        rootDir=self.folderName
        arr = os.listdir(rootDir)
        df = pd.DataFrame(columns=['Asked by','Question','Answered by','Answer'])
        
        asker=''
        question=''
        answerer=''
        answer=''
        gotAnswer=False
        
        # Going through each folder in the provided root folder
        for eachFolder in arr:
            folder= rootDir+'/'+eachFolder
            fileList=os.listdir(folder)
            
            #Going through each word files in the folder
            for eachfile in fileList:
                file=folder+'/'+eachfile
                my_text = docx2txt.process(file)
                alist=my_text.split('\n')
                b=list(filter(lambda a: a != '', alist))
                
                for each in b:
                    if (re.search('^Interviewer.*:',each.strip())):
                        a=each.split(':')
                        if(len(a)==2):
                            asker=a[0].strip()
                            question=a[1].strip()
                    
                    else:
                        aa=each.split(':')
                        if(len(aa)==2):
                            answerer=aa[0].strip()
                            answer=aa[1].strip()
                            gotAnswer=True
                    
                    if(gotAnswer):           
                        df=df.append({'Asked by': asker, 'Question': question, 'Answered by':answerer,'Answer':answer}, ignore_index=True)    
                        gotAnswer=False
                        
        #Final dataframe have 5 columns ['Asked by', 'Question', 'Answered by', 'Answer']              
        return df                                