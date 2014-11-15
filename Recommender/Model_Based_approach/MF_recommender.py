#!/usr/bin/python

from rsvd import RSVD
import numpy as np
import json
import sys 
import random as rnd 
import math
import matplotlib.pyplot as plt

class recommender(object):
	"""class to load the MF model and generate predictions"""

	def __init__(self,dataPath):
		"""load the pretrained model"""
		self.model = RSVD().load(dataPath)
		self.numArtists, self.numFeatures = self.model.u.shape

	def query(self,queryVec,topk,numIter = 800, lr=0.005, regFactor = 0.0001):
		"""generate a prediction based on our model and on the query. 
				- queryVec is a dictionary {artistId:score}
				- topk is the size of the putput prediction
				- lr is the learning rate used for gradient descent 
				- regFactor is the regularization factor 
		"""
		### We fisrt normalize our values
		for artistId in queryVec:
			queryVec[artistId] = float(queryVec[artistId])/math.sqrt(sum([x*x for x in queryVec.values()]))

		### We perform gradient descent computing the error on the values given by the query
		projection = np.random.normal(1./math.sqrt(self.numFeatures), 1., self.numFeatures) #the projection of queryVec on u, estimated using gradient descent
		artist_score = queryVec.items()
		error=[]

		for iter in xrange(numIter):
			rnd.shuffle(artist_score)
			for artistId, targetScore in artist_score:
				err = targetScore - np.dot(projection,self.model.u.transpose())[artistId]
				for k in xrange(self.numFeatures):
					projection[k] += lr*(err*self.model.u[artistId,k]-regFactor*projection[k])
			s = []
			for aId in queryVec:
				s.append((queryVec[aId] - np.dot(projection,self.model.u.transpose())[aId])**2)
			error.append(sum(s))

		plt.plot(error)
		plt.show()

		### using the projection of the query on u obtained by gradient descent we compute the estimated ratings
		reconstruction = np.dot(self.model.u,projection)

		### we sort the ratings
		for artistId in queryVec: #we are not interested in prediction artists from the query 
			reconstruction[artistId] = 0.
		sortedArtistsId = np.argsort(reconstruction)[::-1]
		return sortedArtistsId[:topk],[reconstruction[val] for val in sortedArtistsId[:topk]]


if __name__ == "__main__":

	DATAPATH = "/home/mat/Documents/Git/GroovesharkRecommender/Recommender/Model_Based_approach/saved_model/"
	INV_ARTISTS_LUT_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/inv_playlists_lut.json"
	LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/lexicon.json"

	LEXICON = None
	INV_ARTISTS_LUT = None

	with open(INV_ARTISTS_LUT_PATH,'r') as jsonStream:
		INV_ARTISTS_LUT = json.load(jsonStream)
	with open(LEXICON_PATH,'r') as jsonStream:
		LEXICON = json.load(jsonStream)

	QUERY = {201:10,2434:6,2043:8,1090:8,53:7,2943:5,5291:6,249:5,262:8} 

	print "###############################"
	print "##           QUERY           ##"
	print "###############################"
	for artistId in QUERY:
		print LEXICON["artists"][INV_ARTISTS_LUT[str(artistId)]]["name"],":",QUERY[artistId]

	rec = recommender(DATAPATH)
	answer,scores = rec.query(QUERY,20)

	print "###############################"
	print "##           answer          ##"
	print "###############################"
	for idx,artistId in enumerate(answer):
		print LEXICON["artists"][INV_ARTISTS_LUT[str(artistId)]]["name"],scores[idx]


