# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle

# Preprocessing Tools
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB,BernoulliNB 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# %% [markdown]
# ### Preprocessing
# %% [markdown]
# #### Reading Data

# %%
def get_all_data():
    root = "data/"

    with open(root + "imdb_labelled.txt", "r") as text_file:
        data = text_file.read().split('\n')
         
    with open(root + "amazon_cells_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    return data

whole_data = get_all_data()

# %% [markdown]
# #### Removing punctuations, stopwords, and converting words to root form of words

# %%
# remove punctuatons and stopwords
def form_sentence(word):
    word_blob = TextBlob(word)
    return ' '.join(word_blob.words)

stop_words = stopwords.words('english')
j=0
for i in stop_words:
    if i == 'not' or i == "isn't":
        stop_words.pop(j)
    j+=1

# converting to root form
def normalization(word_list):
        lem = WordNetLemmatizer()
        normalized = []
        for word in word_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized.append(normalized_text)
        return normalized
    
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            tmp = single_data.split("\t")
            tmp[0] = form_sentence(tmp[0][:len(tmp[0])-1]).lower()
            word_tokens = word_tokenize(tmp[0]) 
            tmp[0] = [w for w in word_tokens if not w in stop_words]
            tmp[0] = normalization(tmp[0])
            s=''
            for w in tmp[0]:
                s += w + ' '
            tmp[0]=s
            processing_data.append(tmp)
            
    return processing_data

whole_data = preprocessing_data(get_all_data())


# %%
whole_data

# %% [markdown]
# #### Test-Train Split

# %%
def split_data(data):
    total = len(data)
    #  split with train 85%   
    training_ratio = 0.85
    training_data = []
    evaluation_data = []
    random.shuffle(data)
    

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data

train, val = split_data(whole_data)


# %%
train_text = [data[0] for data in train]
train_result = [ord(data[1])-ord('0') for data in train]
val_text = [data[0] for data in val]
val_result = [ord(data[1])-ord('0') for data in val]

# %% [markdown]
# #### Tfidf vectorization

# %%
vectorizer = TfidfVectorizer()
train_text = vectorizer.fit_transform(train_text)
# val_text = vectorizer.transform(val_text)


# %%


# %% [markdown]
# ### Model and Training

# %%
def training_step(text, result):
    return MultinomialNB().fit(text, result)


# %%
classifier = training_step(train_text, train_result)
result = classifier.predict(vectorizer.transform(["I love this movie!"]))

result

# %% [markdown]
# ### Evaluation

# %%
def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))
  
def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == 1 else "Negative"
    print(text, ":", print_text)


# %%
print_result(analyse_text(classifier, vectorizer, "this is a good movie"))

# %% [markdown]
# #### Running on Validation Data

# %%
def simple_evaluation(evaluation_text, evaluation_result, vectorizer):
    total = len(evaluation_text)
    corrects = 0
    for index in range(total):
        analysis_result = analyse_text(classifier, vectorizer, evaluation_text[index])
        text, result = analysis_result
        corrects += 1 if result[0] == evaluation_result[index] else 0

    return corrects * 100 / total

simple_evaluation(val_text, val_result, vectorizer)

# %% [markdown]
# #### Confusion Matrix

# %%
def create_confusion_matrix(evaluation_text, actual_result, vectorizer):
    prediction_result   = []
    for text in evaluation_text:
        analysis_result = analyse_text(classifier, vectorizer, text)
        prediction_result.append(analysis_result[1][0])
    
    matrix = confusion_matrix(actual_result, prediction_result)
    return matrix

confusion_matrix_result = create_confusion_matrix(val_text, val_result, vectorizer)


# %%
pd.DataFrame(
    confusion_matrix_result, 
    columns=["Negatives", "Positives"],
    index=["Negatives", "Positives"])

