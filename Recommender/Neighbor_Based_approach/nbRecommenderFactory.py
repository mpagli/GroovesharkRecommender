#!/usr/bin/python

import sys
import json
import nbJaccardRecommender as jr
import nbCosineRecommender as cr

class NBRecommenderFactory(object):
    """"""
    @staticmethod
    def get_NBRecommender(distance_function, dataset, idx2name):
        if distance_function.lower() == "cosine":
            return cr.NBCosineRecommender(dataset, idx2name)
        elif distance_function.lower() == "jaccard":
            return jr.NBJaccardRecommender(dataset, idx2name)
        else:
            sys.stderr.write("Unknown distance function:", distance_function)
            sys.exit(1)

if __name__ == "__main__":

    DATA_PATH = "/home/mat/Documents/Git/GroovesharkRecommender/Recommender/Data_processing/f_bow_large_corpus.json"
    with open(DATA_PATH, 'r') as jsonStream:
        DATA = json.load(jsonStream)

    DATASET = DATA["corpus"].values() #transform the dict into a list
    IDX2NAME = DATA["invLexicon"]

    print "\n#playlists:", len(DATASET), "  #artists:", len(IDX2NAME), "\n\n"

    recommender = NBRecommenderFactory.get_NBRecommender('cosine', DATASET, IDX2NAME)

    QUERY = [("9836", 10), ("1817", 5), ("4361", 5), ("7995", 5)]
    for artist_idx, count in QUERY:
        print IDX2NAME[artist_idx], " -> count:", count

    recommendation = recommender.get_recommendation(QUERY)

    print "\n\n"
    for name, score in recommendation:
        print name, " -> score:", score
