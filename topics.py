import sys
import os.path
import json
import gzip

from gensim import corpora
from gensim.models import LsiModel
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer


# Extract single file
def extract(filepath):
    with gzip.open(filepath, 'r') as raw:
        return json.loads(raw.read().decode('ascii'))


# Generator to determine index of current article
def counter():
    current = 0
    while True:
        yield current
        current += 1


# Load texts and titles
def load(text_filepath, titles_filepath):
    return extract(text_filepath), extract(titles_filepath)


# Given a string, perform the following actions:
# 1) Tokenization
# 2) Stop word removal
# 3) Lemmatization (using stemmer)
def clean(text, tokenizer, stops, stemmer):
    tokens = tokenizer.tokenize(text.lower())
    is_valid = lambda x : (x not in stops) and (len(x) > 1)
    return [ stemmer.stem(t) for t in tokens if is_valid(t) ]


# Perform preprocessing on each article
def preprocess(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    stops = set(stopwords.words('german'))
    stemmer = GermanStemmer()
    indexer = counter()

    cleaned = []
    for text in docs:  
        idx = next(indexer)      
        print(f'Processing article {idx}', end='\r')
        cleaned.append(clean(text, tokenizer, stops, stemmer))
    return cleaned


# Get dictionary of terms and document-term matrix
def prepare_corpus(docs):
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [ dictionary.doc2bow(doc) for doc in docs ]
    return dictionary, doc_term_matrix


# Get LSI model
def get_model(docs, num_topics, num_words):
    dictionary, doc_term_matrix = prepare_corpus(docs)
    model = LsiModel(doc_term_matrix, num_topics=num_topics, \
                    id2word=dictionary) 
    print(model.print_topics(num_topics=num_topics, num_words=num_words))
    return model


# Main function
def lsa(num_topics=7, num_words=10):
    docs, titles = load('news_3d_druck_texts.json.gz', \
                        'news_3d_druck_titles.json.gz')
    clean_text = preprocess(docs)
    print('\nGenerating LSA model')
    return get_model(clean_text, num_topics, num_words)


# Run directly from command line
if __name__ == '__main__':
    assert len(sys.argv) == 3
    lsa(int(sys.argv[1]), int(sys.argv[2]))
