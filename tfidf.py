import sys
import os
import math



class tfidf:
    def __init__(self):

        self.documents = dict([])
        self.occurrences = dict({})



    def addDocument(self, name, words):
       
        doc_dict = dict([])
        occurrence = False
        for w in words:
           
            doc_dict[w] = doc_dict.get(w, 0.0) + 1.0
            if not occurrence:
                
                self.occurrences[w] = self.occurrences.get(w, 0.0) + 1.0
                occurrence = True
       
        length = float(len(words))
        for key in doc_dict:
            doc_dict[key] = doc_dict[key] / length
        
        self.documents[name] = doc_dict

  
    def get_similarities(self, word, weights, type='normal', scaling=1.0):
    
        num_docs = len(self.documents)
        for doc, doc_dict in self.documents.items():
            score = 0.0
            if (word in self.documents) and (word in doc_dict):
                if type == 'normal':
                    score += self.normal(word, doc_dict[word], num_docs)
                elif type == 'smooth':
                    score += self.smooth(word, doc_dict[word], num_docs)
                else:
                    print (type, 'is not a valid td-idf function')
                    sys.exit(1)
                score *= scaling
            weights[doc][word] = score



    def normal(self, word, freq, num_docs):
        
        tf = 0.5 + (0.5 * freq)
        idf = math.log(num_docs / self.occurrences[word], num_docs)
        return tf * idf

    def smooth(self, word, freq, num_docs):

        tf = math.log(1 + freq)
        idfN = (num_docs - self.occurrences[word]) / self.occurrences[word]
        idf = math.log(1 + idfN)
        return tf * idf
