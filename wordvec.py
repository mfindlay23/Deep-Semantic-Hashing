import os
import gensim
import word2vec

#Initializes an iterator class that will iterate through all of the docs in the corpus and 
#create a vector for each doc containing vectors of each line containing a vector of each word
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for dname in os.listdir(self.dirname):
            #print(dname)
            if(dname != ".DS_Store"):        
                dname = os.path.join(self.dirname, dname)
                for fname in os.listdir(dname):
                    for line in open(os.path.join(dname, fname)):
                        yield line.split()

#creating the vectors from the address provided below, will read a directory of directories containing the docs
#other directory arrangements will not work in this version
sentences = MySentences('./20news-bydate-train/') # a memory-friendly iterator
list(sentences)
#print(list(sentences))

#passing the vectors into word2vec, dimesion of 256 currently
model = gensim.models.word2vec.Word2Vec(sentences, size=256, sorted_vocab=1)
#model.save('wv.txt') 
print(model)
#print(model.wv.vocab)

#each input prints out the vector for the word, just hit "return" to end, or type a word not in the corpus
word = 'empty_string'
while (word != '\n'):
    word = raw_input("Enter a word: ")
    print(model.wv[word])

#model.build_vocab(sentences)
#print(model)
#print(model.wv.vocab)