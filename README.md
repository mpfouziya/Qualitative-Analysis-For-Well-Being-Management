# Qualitative-Analysis-For-Well-Being-Management

## Overview
Addressing the overall well-being of underserved communities has been a missing part of puzzle in Big Data Science. Every individual should have an equal opportunity to reach their full potential in every sphere of their life, but reality is far behind this. Since so many individuals lack the opportunity to improve their lifestyle, the expulsion of health disparity has been emerged as a major worldwide public health objective. This project focuses on the analysis of the reasons for the prevalence of these problems in a rural community in India and thus preventing those at low cost and high speed. 

To unveil this, the content of the text data that has been obtained by conducting surveys and informal interviews, are analyzed using two different approaches: (i) Using Word2Vec which helps in understanding the relevant related keywords from the data and (ii) A Text Classification approach using a simple Artificial Neural Network (ANN), whose result is then visualized using a WordCloud. Textual data contains abundant qualitative information that are not easy to undergo a statistical analysis unlike quantitative data. The findings say that, for our data, the combination of Text Classification with Word2Vec provides more efficient results than using those modeling approaches individually, as it can find niche topics and associated vocabularies from the interview data. This project report provides an overview on the qualitative research, the techniques that are used to analyze our textual data for obtaining meaningful information, the limitations of those approaches and suggests some possible ways for further study.


## Technical Aspect
This project is divided into two part:
1. Data Preprocessing
   *    Translate and transcribe the data into English language, as the interview data is in Gujarati and Hindi language.
   *	Prepare the training data.
   *	Prepare the transcribed interview data to make it ready for analysis.
   *	Performing advanced simulations on the resulting data:
       *	Using Word2Vec: Convert each and every word in the data into vector format which will then help to identify the semantic similarities between them.
       *	Using a Neural Network for text classification: Classify the responses based on the questions which is already been categorized into different groups and identify             the codewords from each group using a Word Cloud.
   *	Compare the developed models and identify the approach that would help to define our theme “Well Being Management” well.

3. Building the model for extracting the text.
    

## Installation
The Code is written in Python 3.9.1 If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img width="159" alt="postgreSQL" src="https://user-images.githubusercontent.com/37532698/108682128-5f5cba00-7509-11eb-9ab4-2cc02f7971c0.png">](https://www.postgresql.org/) [<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org)  


