# -*- coding: utf-8 -*-
"""intro_to_NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QAT_JepEy8OOttDStgp_5NYt1nolvyUV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/gdrive')

df_hotel=pd.read_csv('/content/gdrive/MyDrive/BDS_practice_data/Hotel_Reviews.csv')
df_hotel.head(20)

"""1 a) before doing 1st step itself the below remove No Negative and No Positive review rows"""



df_hotel.shape

(df_hotel.columns)

col_types=df_hotel.dtypes
col_cat_index=[]
col_cat=[]
col_num=[]
for i in range(len(col_types.index)):
  if col_types[i]=='object':
    col_cat_index.append(i)
    col_cat.append(col_types.index[i])
  else:
    col_num.append(col_types.index[i])

col_cat, col_cat_index

"""Hotel_Address,Negative_Review,Positive_Review,Reviewer_Score"""

indices=[0,6,9,12]
df_subset=df_hotel.iloc[:,indices]
df_subset.head(10)

"""1 a) before doing 1st step itself the below remove No Negative and No Positive review rows"""

pos_index=[]
for i in range(len(df_subset.index)):
  if df_subset.Positive_Review[i]=='No Positive':
    pos_index.append(i)
print(len(df_subset.index))
df_subset.drop(pos_index, inplace=True)

print(len(df_subset.index))

df_subset.reset_index(drop=True,inplace=True)

neg_index=[]
for i in range(len(df_subset.index)):
  if df_subset.Negative_Review[i]=='No Negative':
    neg_index.append(i)
print(len(df_subset.index))
df_subset.drop(neg_index, inplace=True)
df_subset.reset_index(drop=True,inplace=True)

"""1) Merge Negative Review and Positive Review Column as One Column named Review"""

df_subset['Review']=(df_subset.Negative_Review.values)+(df_subset.Positive_Review.values)

df_subset.Review[2]

"""2) Count the number of words in Negative Review and Positive Review Column and make a new dataframe naming it explotaryanalysisdf so your new data frame will have Negative Review and Positive Review and Negative Review Word Count and  PositiveReview Word Count"""

a='manoj subedi'
a.split(' ')

neg_word_count=[]
pos_word_count=[]
for i in df_subset.Negative_Review:
  lst=i.split(' ')
  neg_word_count.append(len(lst))
print(len(neg_word_count))
for i in df_subset.Positive_Review:
  lst=i.split(' ')
  pos_word_count.append(len(lst))
len(pos_word_count)

df_subset['Neg_Review_Word_Count']=neg_word_count
df_subset['Pos_Review_Word_Count']=pos_word_count

df_subset.head(10)

"""3) Count how many reviews are their for each property and make a new column named Total_Reviews in explotaryanalysisdf saying the count of total reviews for each property"""

df_subset.Hotel_Address[0]
total_review=df_subset.Hotel_Address.value_counts()
total_review[' s Gravesandestraat 55 Oost 1092 AA Amsterdam Netherlands']

total_rev_lst=[]
for i in df_subset.Hotel_Address:
  total_rev_lst.append(total_review[i])
df_subset['Count_of_total_review']=total_rev_lst

"""4) make a new column is_bad_review if review_score below 5 give it 0 else 1"""

lst_rev_score=[]
for i in df_subset.Reviewer_Score:
  if i<5:
    lst_rev_score.append(0)
  else:
    lst_rev_score.append(1)
df_subset['is_bad_review']=lst_rev_score
df_subset.head(10)

len(df_subset.index)

len(df_subset.columns)

"""5) count for bad and good reviews"""

df_subset.is_bad_review.value_counts()

"""6) Review Count For negative and positive For each day of week

like how many negative reviews in total for monday for tuesday etc.

similarly for positive use "Review_Date"
"""

df_hotel.Review_Date[0]

from datetime import datetime
date_format = "%m/%d/%Y"
df_hotel['New_Date'] = df_hotel['Review_Date'].apply(lambda x: datetime.strptime(x, date_format))

type(df_hotel['New_Date'][0])

df_hotel['Day']=df_hotel['New_Date'].dt.weekday

df_hotel.Day.min()

pos_rev_per_day=df_hotel.groupby('Day').Positive_Review.count()
pos_rev_per_day.index=['mon','tue','wed','thurs','fri','sat','sun']

plt.bar(pos_rev_per_day.index, pos_rev_per_day.values)
plt.xlabel('WeekDay')
plt.ylabel('Positive_rev_count')

neg_rev_per_day=df_hotel.groupby('Day').Negative_Review.count()
neg_rev_per_day.index=['mon','tue','wed','thurs','fri','sat','sun']
neg_rev_per_day

plt.bar(neg_rev_per_day.index, neg_rev_per_day.values)
plt.xlabel('WeekDay')
plt.ylabel('Negative_rev_count')

"""3 models using random forest or any algorithm

1) TFIDF and label   to do classification

2) With Count vectorizer to do classification

3) With Feature engineering like we did
1) number of words in total, Number of words in negative review, Number of words in positive review,total reviews for each property and then add compound value from SentimentIntensityanalyser for both negative review and positive review separately

1) TF-IDF vectorizer and using random forest classification model
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
X = vectorizer.fit_transform(df_subset.Review)

# Print the feature names (words)
#print(vectorizer.get_feature_names())

# Print the TF-IDF matrix
print(X.toarray())

X.shape

y=df_subset.is_bad_review.values

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2, test_size=0.25)

#apply random forest

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=30, random_state=1)
rf_model.fit(X_train, y_train)

y_pred=rf_model.predict(X_test)

from sklearn.metrics import f1_score
f1_score(y_test,y_pred)

"""2) Count vectorizer and apply random forest model"""

from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the text data
Xc = vectorizer.fit_transform(df_subset.Review)

# Print the feature names (words)
#print(vectorizer.get_feature_names())

# Print the document-term matrix
#print(X.toarray())

Xc.shape

from sklearn.model_selection import train_test_split
Xc_train, Xc_test, y_train, y_test = train_test_split(Xc,y, random_state=3, test_size=0.25)

#apply random forest

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=30, random_state=2)
rf_model.fit(Xc_train, y_train)

yc_predict=rf_model.predict(Xc_test)

from sklearn.metrics import accuracy_score, recall_score
print(f1_score(y_test,yc_predict))
print(accuracy_score(y_test,yc_predict))

"""add compound value from SentimentIntensityanalyser for both negative review and positive review separately"""

import nltk

#download the lexicon dictnoary
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance of the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment intensity analysis
# sentence=['I like this motel very much','The room was very dirty. I do not like the hotel']
# sentiment_scores = sia.polarity_scores(sentence) ##does not allow list or any other combined files type

# Get the compound sentiment score
compound_positive=[]
for i in df_subset.Positive_Review:
  sentiment_score=sia.polarity_scores(i)
  compound_positive.append(sentiment_score['compound'])

# Print the compound sentiment score
print(len(compound_positive))

# Get the compound sentiment score
compound_negative=[]
for i in df_subset.Negative_Review:
  sentiment_score=sia.polarity_scores(i)
  compound_negative.append(sentiment_score['compound'])

# Print the compound sentiment score
print(len(compound_positive))

df_subset['compound_score_neagtive']=compound_negative
df_subset['compound_score_positive']=compound_positive



"""### NLP procedure

bringing all the words to lowercase

removing unwanted characters in text (punctuation)

removing useless stopwords like 'the','you','your','my'

tokenization, getting the list of individual words from the sentence

Stemming or lemmatization transform every word into their root form

Two parts of embedding or numerical characterization

1) TF-IDF, count vectorization do not need tokenization

2) models like bert,Doc2Vec need tokenization and removing of unwanted characters

4) Use Sentence Transformer for classification using randomForest
"""

## install sentence transformers module
!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
# BERT covered for 100 languages different models you can read about them

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = list(df_subset.Review)

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding.shape)
#     print("")

print(embeddings.shape)

embeddings.shape

"""5) use clean_text function or preprocessing we discussed create TF-IDF and pass to randomforest  to compare results

Preprocessing of text...cleaning & lemmatization
"""

import nltk

##tokenizer
from nltk.corpus.reader.tagged import word_tokenize

##punctuation
import string
nltk.download('punkt')

## stop words
nltk.download('stopwords') ##non important and redundant words in english like the, in , you , your, etc
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

##lemmatize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer ## lemmatize the word based on context,,, eg. caring into care such that
lemmatizer=WordNetLemmatizer()

from nltk.corpus.reader.tagged import word_tokenize
def clean_token(text):
  text_lowered=text.lower()
  #tokenize the sentence
  text_token=word_tokenize(text_lowered)
  #remove punctuation
  text_token_wo_punc= [i for i in text_token if i not in string.punctuation]
  #remove the stop words
  text_token_wo_stopwords=[j for j in text_token_wo_punc if j not in stop_words]
  #lemmatize the text
  text_token_lemmatized=[lemmatizer.lemmatize(k) for k in text_token_wo_stopwords]
  clean_text=' '.join(text_token_lemmatized)
  return clean_text

sentence='Do you like to 30948 !!! sleep in the couch? Oh! really@@ I have sleeping on couch yesterday. Did you slept well?'
#sentence = "The dogs are barking loudly in the park."
clean_token(sentence)

## applying clean_token function to the review column
df_subset['Clean_Review']=df_subset.Review.apply(clean_token)

df_subset.head(10)

## get TF-IDF for the new Clean_review column
from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
X = vectorizer.fit_transform(df_subset.Review)

X.shape

y=df_subset.is_bad_review.values

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=3, train_size=0.75)

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=30, random_state=1)
rf_model.fit(X_train, y_train)
y_rf2_pred=rf_model.predict(X_test)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
print('f1 score =', f1_score(y_test, y_rf2_pred))
print('accuracy score =', accuracy_score(y_test, y_rf2_pred))
print('precision score =', precision_score(y_test, y_rf2_pred))
print('recall score =', recall_score(y_test, y_rf2_pred))
print(confusion_matrix(y_test, y_rf2_pred))