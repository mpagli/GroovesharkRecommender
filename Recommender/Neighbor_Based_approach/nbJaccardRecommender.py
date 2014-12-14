import numpy as np
import nbRecommender

class NBJaccardRecommender(nbRecommender.NBRecommender):
    """implement a NB recommender system using the jaccard distance function"""

    def __init__(self, dataset, idx2name):
        """Input:
            - dataset: a list of bag of words representation (list of tuples (wordIdx,count))
            - idx2name: a dictionary mapping the idx of the artist in the system to his name.
        Since we are using jaccard distance we can transform the dataset into a list of sets.
        """
        self.dataset = dataset
        self.idx2name = idx2name
        self.format_dataset()

    @staticmethod
    def bow_to_set(bow):
        """input:
            - a list of tuples (wordIdx,count)
           output:
            - a set of wordIdx
        """
        return set([int(word_count[0]) for word_count in bow])

    def format_dataset(self):
        """transform the dataset in a list of sets"""
        for idx in xrange(len(self.dataset)):
            self.dataset[idx] = NBJaccardRecommender.bow_to_set(self.dataset[idx])

    def get_recommendation(self, query, top_k=20):
        """from a bow representation of a playlist, return a recommendation of size top_k.
            - query: a list of tuples (word,count)
            - top_k: the size of the recommendation
        return a list of tuples (artist_name, similarity_score)
        """
        query = NBJaccardRecommender.bow_to_set(query)
        artists_score = [0.]*len(self.idx2name) #one score per artist
        for playlist_set in self.dataset:
            distance = NBJaccardRecommender.jaccard_distance(playlist_set, query)
            for artist_id in playlist_set:
                artists_score[artist_id] += distance
        for artist_idx in query:
            artists_score[artist_idx] = 0. #we don't want to recommend artists in the query
        sorted_idx = np.argsort(artists_score)[::-1]
        sorted_idx = sorted_idx[:top_k]
        return [(self.idx2name[str(artist_idx)], artists_score[artist_idx]) for artist_idx in sorted_idx]

    @staticmethod
    def jaccard_distance(set_a, set_b):
        """Return the jaccard distance between the two sets: 
                     distance(A,B) = |A and B| / |A or B|
        """
        return float(len(set_a & set_b)) / len(set_a|set_b)