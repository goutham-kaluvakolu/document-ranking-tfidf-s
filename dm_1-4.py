# -*- coding: utf-8 -*-
import os
import math
import copy
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stop_words=stopwords.words('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

"""Def all reusable functions"""

def iniwordstotokens(words): # converts a list of words to list of stemmed words without stopwords
  filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
  return filtered_words

def wordstotokens(sentence): # converts a senetnce to list of stemmed words without stopwords
  words=sentence.split(" ")
  return iniwordstotokens(words)

def wordtotoken(word):
    return stemmer.stem(word)

def getWordcount(words): #creates a dic with each terms frequency in the sentence
  dic={}
  for word in words:
    if word in dic:
      dic[word]+=1
    else:
      dic[word]=1
  return dic

directory_path = "US_Inaugural_Addresses"
global_dic={}

'''
preprocess() creates a global valiable gloabl_dic which has a structure :

 global_dic={
  
       file_name:[[all terms],

      {[key:term]:[value:frequncy in that doc]}

      {[key:term],[value:tf-idf-weighted]}]
       
  }'''

def preprocess():
  corpusroot = './US_Inaugural_Addresses'
  for filename in os.listdir(corpusroot):
      if filename.startswith('0') or filename.startswith('1'):
          file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
          doc = file.read()
          file.close() 
          doc = doc.lower()
          words = tokenizer.tokenize(doc)
          tokens=iniwordstotokens(words)
          global_dic[filename]=[tokens,getWordcount(tokens)]

df=0
def get_global_idf():  # creates a key val pair, keys are all the unique words in the courpus and values as idf.
  idf_dic={}
  temp=[]
  for key,val in global_dic.items():
    temp.extend(val[0])
  temp = set(temp)
  temp=sorted(temp)
  for i in temp:
    df=0
    for key,val in global_dic.items():
      if i in val[0]:
        df+=1
    if df==0:
      idf_dic[i]=[-1,df]
    else:
      a=math.log10(15 / (df))
      idf_dic[i]=[a,df]
  return idf_dic

def tfidf_vector():
  for file,val in global_dic.items():
    dic={}
    for key,value in val[1].items():  #val[1] in the global_dic holds all the terms and their frequney in the doc eg {british:2}
      tf_w=1+math.log10(value)
      dic[key]=tf_w*idf_dic[key][0]
    global_dic[file].append(dic)

def normalize_tfidf_in_global_dic(global_dic): #returns normalized tf-idf stored in global_dic created from tfidf_vector()
    normalized_global_dic = copy.deepcopy(global_dic)  # Create a deep copy of the original dictionary
    
    for file_name, data in normalized_global_dic.items():
        tfidf_dict = data[2]  # Assuming TF-IDF weights are in the third element of the data
        doc_length = math.sqrt(sum(value ** 2 for value in tfidf_dict.values()))
        # Normalize TF-IDF weights
        for term, tfidf_weight in tfidf_dict.items():
            normalized_tfidf = tfidf_weight / doc_length
            tfidf_dict[term] = normalized_tfidf
            
    return normalized_global_dic  # Return the new dictionary with normalized TF-IDF values

preprocess()
idf_dic=get_global_idf()
tfidf_vector()
normalized_global_dic = normalize_tfidf_in_global_dic(global_dic) #normalized_global_dic has normalized tf-idf vextor for each file in its 2 index

def getidf(str):
  str=wordtotoken(str)
  dtf=0
  for key,val in global_dic.items():
    if str in val[1]:
      dtf+=1
  if dtf >0:
    return math.log10(15 / (dtf))
  else:
    return -1

def getweight(filename,word):
  token=wordtotoken(word)
    # normalized_global_dic contains the dictionary with normalized TF-IDF values.
  if token in normalized_global_dic[filename][2]:
    return normalized_global_dic[filename][2][token]
  else:
    return 0

def get_tf_weight(tf):
  if tf==0:
    return 0
  else:
    return 1+math.log10(tf)

def query(words):
  tk=wordstotokens(words)
  max=[0,-1]
  #  for all documents
  t=[]
  for file,val in global_dic.items():
    # get the query , doucmnet unique terms
    keys_dict1 = set(val[1].keys())
    keys_dict2 = set(tk)

  # Find the unique keys combined
    unique_keys_combined = keys_dict1.union(keys_dict2)

  # find the tf for the query and the doc
    query_dic={}
    doc_dic={}
    for word in unique_keys_combined:
      count=0
      for tk_word in tk:
        if word==tk_word:
          count+=1
      query_dic[word]=count
      if word in val[1]:
        doc_dic[word]=val[1][word]
      else:
        doc_dic[word]=0

  # get tf-weighted for query and doc
    # query-tf-weighted
    for key,count in query_dic.items():
      query_dic[key]=[count,get_tf_weight(count)]

    # doc-tf-weighted
    for key,count in doc_dic.items():
      doc_dic[key]=[count,get_tf_weight(count)]

  # get tf idf for the doc
    for i in unique_keys_combined:
      if i in idf_dic:
        doc_dic[i].append(idf_dic[i][0])
      else:
        doc_dic[i].append(math.log10(15))

  # get weights
    for key,val in doc_dic.items():
      # print(key,val)
      doc_dic[key].append(val[1]*val[2])

  # normalize both
    sum=0
    for key,val in query_dic.items():
      sum+=val[1]*val[1]
    normalizer=1/(math.sqrt(sum))
    for key,val in query_dic.items():
      query_dic[key].append(val[1]*normalizer)
    sum=0
    for key,val in doc_dic.items():
      sum+=val[3]*val[3]
    normalizer=1/(math.sqrt(sum))
    for key,val in doc_dic.items():
      doc_dic[key].append(val[3]*normalizer)

  # dot product
    simi=0
    for (key1, value1), (key2, value2) in zip(query_dic.items(), doc_dic.items()):
      simi+=value1[2]*value2[4]
    if simi>max[1]:
      max=[file,simi]
  return (max[0],max[1])

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")

print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))

print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))

"""
GLOBAL_DIC holds is a dictionary
of format

 global_dic={
  
       file_name:[[all terms],

      {[key:term]:[value:frequncy in that doc]}

      {[key:term],[value:tf-idf-weight]}
       
  }

IDF_DIC holds idf for all the terms in the corpus

   idf_dic={
  
       Term:[idf,df]
      
  }

QUERY_DIC holds is a dictionary
of format

 query_dic={
  
       Term:[<frequency in query> <tf-weight> <cos-norm>]

  }

DOC_DIC holds idf for all the terms in the corpus

   doc_dic={
  
       Term:[<tf> <tf-w> <idf> <tf-idf> <cos-norm>]
      
  }
"""

