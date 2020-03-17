#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np

data = pd.read_csv("./20news-bydate/matlab/train.data", header=None, delimiter=r'\s+')
labels = pd.read_csv("./20news-bydate/matlab/train.map", header=None, delimiter=r'\s+')
id = pd.read_csv("./20news-bydate/matlab/train.label", header=None, delimiter=r'\s+')
vocab = pd.read_csv("./vocabulary.txt", sep='\t', names=['vocabulary'])
vocab = vocab.reset_index()
vocab['index']+=1
data.columns = ['docidx', 'wordidx', 'count']
labels.columns = ['docname', 'labelid']
id = id.reset_index()
id.columns = ['docidx', 'labelid']
id.docidx+=1
total_features, a = 1000, 0.003


# In[19]:


stopWords = [
"a", "about", "above", "across", "after", "afterwards", 
"again", "all", "almost", "alone", "along", "already", "also",    
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", 
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]


# In[20]:


stop_words = pd.DataFrame(stopWords)
vocab_filtered = [w for w in vocab.vocabulary if not w in stopWords]
vocab_filtered = pd.DataFrame(vocab_filtered)
vocab_filtered.columns = ['vocabulary']
vocab_filtered = pd.merge(vocab, vocab_filtered, on = ['vocabulary'])
stop_word_indices = set(vocab['index'].to_list())-set(vocab_filtered['index'].to_list())
print(stop_word_indices)


# In[21]:


labeled_data = pd.merge(data, id, on=['docidx'])
labeled_data = labeled_data[labeled_data['wordidx'].isin(vocab_filtered['index'].values)]
bag_of_words = labeled_data[['wordidx', 'count']]
bag_of_words = bag_of_words.groupby(['wordidx']).sum()
bag_of_words = bag_of_words.sort_values(by=['count'],ascending=False)
bag_of_words = bag_of_words.reset_index()
bag_of_words = bag_of_words[0:total_features]
labeled_data = labeled_data[labeled_data['wordidx'].isin(bag_of_words.wordidx.values)]
labeled_data = labeled_data.reset_index()
del labeled_data['index']
#print(labeled_data)
unique_wordidx = bag_of_words.wordidx.values


# In[22]:


labeled_docs = labeled_data[['docidx','labelid']]
labeled_docs = labeled_docs.drop_duplicates()
prior_info = labeled_docs.labelid.value_counts().rename_axis().reset_index(name='documents')
prior_info.documents/=prior_info.documents.sum()
print(prior_info)


# In[23]:


import numpy as np
import dirichlet
from scipy.special import gammaln,softmax,gamma

h= total_features

## to get all the words after pivoting the dataframe
for i in range(1,21):
    z = 11270+i
    df2= pd.DataFrame({'docidx': [z]*h, 'wordidx':[x for x in unique_wordidx], 'count':[0]*h, 'labelid':[i]*h})
    labeled_data = labeled_data.append(df2, ignore_index=True)
coeff = []

for i  in range(1,21):
    p = labeled_data[labeled_data['labelid']==i]
    d = p.pivot(index = 'docidx', columns = 'wordidx', values = 'count')
    d = d.fillna(0)
    d['sum'] = d.sum(axis=1)
    d+=a
    d = d.loc[:,:].div(d['sum']+a*(h-1), axis=0)
    d = d[0:-1]
    del d['sum']
    f = dirichlet.mle(d.values)  ## mle estimation for dirichlet distribution
    #f = softmax(f)
    #f/= f.sum()
    l = gammaln(f.sum()) - gammaln(f).sum()
    coeff.append(l)
    print(i, gammaln(f.sum())-gammaln(f).sum())
    c = pd.DataFrame({'wordidx':d.columns.tolist(), i:f})
    c = c.sort_values(by=['wordidx'])
    #total_list = [x for x in range(1,61189)]
    #remaining_ind = set(total_list)-set(d.columns.to_list())
    #print(list(remaining_ind))
    #x = pd.DataFrame({'wordidx':list(remaining_ind), i:[0]*len(remaining_ind)})
    #c = c.append(x, ignore_index=True)
    if i==1:
        parameters = c
    else:
        parameters = pd.merge(parameters, c, on=['wordidx'])
        
#print(parameters) 


# In[24]:


parameters = parameters.sort_values(by=['wordidx'])
#parameters
coeff = pd.DataFrame(coeff)
coeff.columns = ['coefficients']
#print(coeff)
coeff.to_csv("./beta_coefficients.csv")
word_indices = pd.DataFrame(parameters.wordidx.values)
word_indices.columns = ['wordidx']
word_indices.to_csv("./word_indices.csv")
g = parameters
del g['wordidx']
g = g.transpose()
g.columns = [x for x in range(1,total_features+1)]
#print(g)
g.to_csv("./dirichlet _parameters.csv", index=False)


# In[25]:


test_data = pd.read_csv("./20news-bydate/matlab/test.data", header=None, delimiter=r'\s+', names=['docidx','wordidx','count'])
test_data = test_data[test_data['wordidx'].isin(unique_wordidx)]
test_id = pd.read_csv("./20news-bydate/matlab/test.label", header=None, delimiter=r'\s+')
test_id = test_id.reset_index()
test_id.columns = ['docidx', 'labelid']
test_id.docidx+=1
test_labeled_data = pd.merge(test_data, test_id, on=['docidx'])
#print(test_labeled_data.docidx.max())
for i in range(1,21):
    z = 7510+i
    df2= pd.DataFrame({'docidx': [z]*h, 'wordidx':[x for x in unique_wordidx], 'count':[0]*h, 'labelid':[i]*h})
    test_labeled_data = test_labeled_data.append(df2, ignore_index=True)

prior_info = prior_info.sort_values(by=['index'])
prior_info = np.log(prior_info['documents'].values)
#print(prior_info, coeff.values)
coeff = pd.read_csv("./beta_coefficients.csv")
#print(coeff['coefficients'].values)
h = total_features
correctly_estimated = 0
total_test_docs = test_labeled_data.docidx.nunique()

for i  in range(1,21):
    p = test_labeled_data[test_labeled_data['labelid']==i]
    d = p.pivot(index = 'docidx', columns = 'wordidx', values = 'count')
    d = d.fillna(0)
    d['sum'] = d.sum(axis=1)
    d+=a
    d = d.div(d['sum']+a*(h-1), axis=0)
    d = d[0:-1]
    del d['sum']
    res = np.log(d.values)@(g.transpose()-1)
    print("res shape:",res.shape)
    for j in range(res.shape[0]):
        if np.argmax(res.loc[j].values+prior_info+coeff['coefficients'].values)==i-1:
            correctly_estimated+=1
            
print("test_accuracy: ",correctly_estimated/total_test_docs)

