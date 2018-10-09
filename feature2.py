import os
import sys
import string
import nltk
from tfidf import tfidf
import operator


datafile = 'dataset2.csv'



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
    # sort set into list
    return sorted(features)


def generate_weights(documents, lexicon):
   
    
    weights = dict()
    m = tfidf()
    print('Adding documents for TF-IDF...')
    for i, document in enumerate(documents):
        m.addDocument(i, document['words']['title']+document['words']['body'])
        weights[i] = dict()
    
    print('Generating weight scores for words; This WILL take time...')
    for word in lexicon['title'] & lexicon['body']:
        m.get_similarities(word, weights, 'smooth', 1.25)
    for word in lexicon['title'] - lexicon['body']:
        m.get_similarities(word, weights, 'smooth', 1.1)
    for word in lexicon['body'] - lexicon['title']:
        m.get_similarities(word, weights, 'smooth')
    return weights


def generate_dataset(documents, lexicon):
    """ function: generate_dataset
        --------------------------
        select features from @lexicon for feature vectors
        generate dataset of feature vectors for @documents

        :param documents: list of well-formatted, processable documents
        :param lexicon:   list of word stems for selecting features
    """
    print ('\nGenerating dataset @', datafile)
    weights = generate_weights(documents, lexicon)


    print ('Selecting features for the feature vectors @', datafile)
    features = select_features(weights)

   
    print ('Writing feature vector data @', datafile)
    generate_csv(documents, features, weights)
    print ('Finished generating dataset @', datafile)
