import os
import sys
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import operator


datafile = 'dataset3.csv'



def generate_csv(documents, features, weights):

    dataset = open(datafile, "wb")
    dataset.write(b'id\t')
    for feature in features:
        dataset.write(bytes(feature.encode('ascii','ignore')))
        dataset.write(b'\t')
    dataset.write(b'class-label:topics\t')
    dataset.write(b'class-label:places\t')
    dataset.write(b'\n')
   
    for i, document in enumerate(documents):
       
        dataset.write(bytes(str(i).encode('ascii','ignore')))
        dataset.write(b'\t')
     
        for feature in features:
            dataset.write(bytes(str(weights[i][feature]).encode('ascii','ignore')))
            dataset.write(b'\t')
      
        dataset.write(bytes(str(document['topics']).encode('ascii','ignore')))
        dataset.write(bytes(str(document['places']).encode('ascii','ignore')))
        dataset.write(b'\n')
    dataset.close()

def select_features(weights):
   
    features = set()
    for doc, doc_dict in weights.items():
        top = dict(sorted(doc_dict.items(), key=operator.itemgetter(1), reverse=True)[:5])
        for term, score in top.items():
            if score > 0.0:
                features.add(term)
   
    return sorted(features)



def generate_weights(documents):
  
 
    token_dict = dict([])
    for i, document in enumerate(documents):
        token_dict[i] = str(b' '.join(document['words']['title'] + document['words']['body']))
    
    tfidf = TfidfVectorizer()
    weights = tfidf.fit_transform(token_dict.values())
    features = tfidf.get_feature_names()
    return features, weights



def generate_dataset(documents, lexicon):
 
    print ('\nGenerating dataset @', datafile)
    words, weights = generate_weights(documents)

    
    weight_array = weights.toarray()
    weight_dict = dict([])
    for i, row in enumerate(weight_array):
        weight_dict[i] = dict([])
        for j, word in enumerate(words):
            weight_dict[i][word] = weight_array[i][j]

    
    print ('Selecting features for the feature vectors @', datafile)
    features = select_features(weight_dict)

    
    print ('Writing feature vector data @', datafile)
    generate_csv(documents, features, weight_dict)
    print ('Finished generating dataset @', datafile)
