import numpy as np
import nbRecommender

class NBCosineRecommender(nbRecommender.NBRecommender):
    """implement a NB recommender system using the cosine distance function"""
    def __init__(self, dataset, idx2name):
        """Input:
            - dataset: a list of bag of words representation (list of tuples (wordIdx,count))
            - idx2name: a dictionary mapping the idx of the artist in the system to his name.
        """
        self.dataset = dataset
        self.idx2name = idx2name

    def bow_to_vector(self, bow):
        """transforms a bow representation into a vector representation where the index represent
        the Id of the artist
        """
        vec = np.array([0.]*len(self.idx2name))
        for artist_id, count in bow:
            vec[int(artist_id)] = float(count)
        return vec

    def cosine_distance(self, vec, bow_b):
        """return the cosine distance between two one vector and one representations
                            distance(A, B) = A.B/(|A|.|B|)
        """
        vec_b = self.bow_to_vector(bow_b)
        return np.dot(vec, vec_b)/(np.linalg.norm(vec)*np.linalg.norm(vec_b))

    def get_recommendation(self, query, top_k=20):
        """from a bow representation of a playlist, return a recommendation of size top_k.
            - query: a list of tuples (word,count)
            - top_k: the size of the recommendation
        return a list of tuples (artist_name, similarity_score)
        """
        query_vec = self.bow_to_vector(query)
        artists_score = [0.]*len(self.idx2name) #one score per artist
        for playlist_bow in self.dataset:
            distance = self.cosine_distance(query_vec, playlist_bow)
            for artist_id, count in playlist_bow:
                artists_score[artist_id] += distance*count
        for artist_idx, _ in query:
            artists_score[int(artist_idx)] = 0. #we don't want to recommend artists in the query
        sorted_idx = np.argsort(artists_score)[::-1]
        sorted_idx = sorted_idx[:top_k]
        return [(self.idx2name[str(artist_idx)], artists_score[artist_idx]) for artist_idx in sorted_idx]