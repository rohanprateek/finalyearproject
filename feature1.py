import os
import sys
import string
import nltk
from tfidf import tfidf
import operator



datafile = 'dataset1.csv'

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
    """ function: select_features
        -------------------------
        generated reduced feature list for vector generation

        :param weights: dictionary from results of the tf-idf calculations
        :returns: sorted list of terms representing the selected features
    """
    features = set()
    for doc, doc_dict in weights.items():
        top = dict(sorted(doc_dict.items(), key=operator.itemgetter(1), reverse=True)[:5])
        for term, score in top.items():
            if score > 0.0:
                features.add(term)
    # sort set into list
    return sorted(features)



def generate_weights(documents, lexicon):
    """ function: generate_weights
        --------------------------
        perform tf-idf to generate importance scores for words in documents

        :param document: list of documents to use in calculations
        :returns: dictionary of dictionaries: {"id_" : {"word" : score,...}}
    """
    
    weights = dict()
    m = tfidf()
    print('Adding documents for TF-IDF...')
    for i, document in enumerate(documents):
        m.addDocument(i, document['words']['title']+document['words']['body'])
        weights[i] = dict()
    
    print('Generating weight scores for words; This WILL take time...')
    for word in lexicon['title'] | lexicon['body']:
        # UNCOMMENT FOR SANITY
        # print('Generating weights for word:', word)
        m.get_similarities(word, weights)
    return weights


def generate_dataset(documents, lexicon):
   
    print ('\nGenerating dataset @', datafile)
    weights = generate_weights(documents, lexicon)

 
    print ('Selecting features for the feature vectors @', datafile)
    features = select_features(weights)

   
    print ('Writing feature vector data @', datafile)
    generate_csv(documents, features, weights)
    print ('Finished generating dataset @', datafile)
