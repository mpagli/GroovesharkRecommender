#!/usr/bin/python

import json
import numpy as np
import gensim, gensim.models.lsimodel
from gensim import models
import sys

def iter_documents(corpus,lexicon,threshold_artistPop,threshold_artistCount):
    """Iterate over all playlists, yielding a document (=list of artistId with duplicates) at a
    time.
    """
    for playlist_id in corpus:
        acc = {}
        token_list = []
        for song_id in corpus[playlist_id]:
            artist_id = lexicon["songs"][song_id]["artistId"]
            if lexicon["artists"][artist_id]["count"] < threshold_artistPop:
                continue
            if artist_id in acc:
                acc[artist_id] += 1
            else:
                acc[artist_id] = 1
            token_list.append(artist_id)
        if len(acc) >= threshold_artistCount:
            yield token_list

def save_Idx2ArtistName_LUT(corpus, lexicon, threshold_artistPop, threshold_artistCount):
    """save the dictionary in LSI_LUT.txt"""
    LUT = {}
    for tokens in iter_documents(corpus, lexicon, threshold_artistPop, threshold_artistCount):
        for artist_id in tokens:
            if artist_id not in LUT:
                LUT[artist_id] = lexicon["artists"][artist_id]["name"]
    fstream = open("LSI_LUT.txt", 'w')
    for artist_id, name in LUT.items():
        try:
            fstream.write(name+": "+artist_id+"\n")
        except:
            #print "WARNING: Encoding failure."
            pass
    fstream.close()

class MyCorpus(object):
    """corpus formated for gensim"""
    def __init__(self, corpus, lexicon, threshold_artistCount, threshold_artistPop):
        self.dictionary = gensim.corpora.Dictionary(iter_documents(corpus, lexicon,\
        threshold_artistPop, threshold_artistCount))
        self.corpus = corpus
        self.lexicon = lexicon
        self.threshold_artistPop = threshold_artistPop
        self.threshold_artistCount = threshold_artistCount

    def __iter__(self):
        for tokens in iter_documents(self.corpus, self.lexicon, self.threshold_artistPop,\
            self.threshold_artistCount):
            yield self.dictionary.doc2bow(tokens)

if __name__ == "__main__":

    CORPUS_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/corpus.json"
    LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/lexicon.json"

    CORPUS = None   #contain all the playlists as lists of songId
    LEXICON = None  #contain all the data for each artist/song

    T_ARTIST_POP = 200 
    T_ARTIST_COUNT = 3
    NUM_TOPICS = 25

    ### LOADING THE DATA
    print "\nLoading corpus from "+CORPUS_PATH+" ...",
    sys.stdout.flush()
    with open(CORPUS_PATH, 'r') as jsonStream:
        CORPUS = json.load(jsonStream)
    print "ok"
    print "\nLoading lexicon from "+LEXICON_PATH+" ...",
    sys.stdout.flush()
    with open(LEXICON_PATH, 'r') as jsonStream:
        LEXICON = json.load(jsonStream)
    print "ok"

    save_Idx2ArtistName_LUT(CORPUS,LEXICON, threshold_artistPop=T_ARTIST_POP, threshold_artistCount=T_ARTIST_COUNT)

    ### FORMAT THE DATASET
    gensimCorpus = MyCorpus(CORPUS,LEXICON, threshold_artistPop=T_ARTIST_POP, threshold_artistCount=T_ARTIST_COUNT)
    tfidf = models.TfidfModel(gensimCorpus)

    ### PERFORM LSI
    print "LSI ...",
    sys.stdout.flush()
    lsi = gensim.models.lsimodel.LsiModel(tfidf[gensimCorpus], num_topics=NUM_TOPICS)
    print "ok"

    QUERY_ = [80911]*3+[676]*5+[3531]*3+[50]*5+[673]*1+[836]*5
    QUERY = gensimCorpus.dictionary.doc2bow([str(id) for id in QUERY_])

    print "###############################"
    print "##           QUERY           ##"
    print "###############################"
    for artistId in QUERY_:
        print LEXICON["artists"][str(artistId)]["name"]#,":\t",count
    print ""

    proj = [x[1] for x in lsi[QUERY]]
    print proj
    artistRating = np.dot(lsi.projection.u, np.array(proj))
    print "artistRating ", artistRating.shape
    for artistId, count in QUERY:
        artistRating[artistId] = -1e10
    prediction = np.argsort(artistRating)[::-1]
    print "prediction  ", prediction.shape

    print "###############################"
    print "##           answer          ##"
    print "###############################"
    for k in xrange(40):
        artistId = prediction[k]
        print LEXICON["artists"][gensimCorpus.dictionary.get(artistId)]["name"], ":\t",artistRating[artistId]