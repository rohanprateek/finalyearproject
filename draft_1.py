import os
import sys
import string
import nltk
import threading 
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup

import feature1
import feature2
import feature3



def init_document():
    document = { 'topics' : [], 'places' : [], 'words' : dict([]) }
    document['words']['title'] = []
    document['words']['body']  = []
    return document

def populate_class_label(document, article):
    for topic in article.topics.children:
        document['topics'].append(topic.text.encode('ascii', 'ignore'))
    for place in article.places.children:
        document['places'].append(place.text.encode('ascii', 'ignore'))

def populate_word_list(document, article):
    text = article.find('text')
    title = text.title
    body = text.body
    if title != None:
        document['words']['title'] = tokenize(title.text)
    if body != None:
        document['words']['body'] = tokenize(body.text)

def tokenize(text):
    ascii = text.encode('ascii', 'ignore')
    no_digits = ascii.replace('\d+'.encode('ascii','ignore'),bytes('None'.encode('ascii','ignore')))
    no_punctuation = no_digits.replace('[\w\s]'.encode('ascii','ignore'),bytes('None'.encode('ascii','ignore')))
    tokens = nltk.word_tokenize(no_punctuation.decode('ascii','ignore'))
    no_stop_words = [w for w in tokens if not w in stopwords.words('english')]
    eng = [y for y in no_stop_words if wordnet.synsets(y)]
    lemmas = []
    lmtzr = WordNetLemmatizer()
    for token in eng:
        lemmas.append(lmtzr.lemmatize(token))
    stems = []
    stemmer = PorterStemmer()
    for token in lemmas:
        stems.append(stemmer.stem(token).encode('ascii','ignore'))
    terms = [x for x in stems if len(x) >= 4]
    return terms

def generate_document(text):
    document = init_document()
    populate_class_label(document, text)
    populate_word_list(document, text)
    print(document)
    return document



def generate_tree(text):
     return BeautifulSoup(text, "html.parser")



def parse_documents():
    documents = []
    for file in os.listdir('C:\\Users\\Rohan Prateek\\data'):
        data = open(os.path.join(os.getcwd(), "C:\\Users\\Rohan Prateek\\data", file), 'r')
        text = data.read()
        data.close()
        tree = generate_tree(text.lower())
        for reuter in tree.find_all("reuters"):
            document = generate_document(reuter)
            documents.append(document)
        print("Finished extracting information from file:", file)
    return documents



def generate_lexicon(documents):
    lexicon = { 'title' : set(), 'body' : set() }
    for document in documents:
        for term in document['words']['title']:
            lexicon['title'].add(term)
        for term in document['words']['body']:
            lexicon['body'].add(term)
    return lexicon



def main(argv):
    print('Generating document objects. This may take some time...')
    documents = parse_documents()

    print('Document generation complete. Building lexicon...')
    lexicon = generate_lexicon(documents)

    feature1.generate_dataset(documents, lexicon)
    feature2.generate_dataset(documents, lexicon)
    feature3.generate_dataset(documents, lexicon)
    
    print(len(lexicon['title']))
    print(len(lexicon['body']))

    
if __name__ == "__main__":
    main(sys.argv[1:])
