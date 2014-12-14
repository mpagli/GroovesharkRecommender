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
		#self.model = RSVD().load(dataPath)
		self.Q = np.load(dataPath).transpose()
		self.numArtists, self.numFeatures = self.Q.shape
		print self.numArtists, self.numFeatures

	def query(self,queryVec,artistsMean,artistsPlaylistCount,topk,numIter = 800, lr=0.0005, regFactor = 0.0):
		"""generate a prediction based on our model and on the query. 
				- queryVec is a dictionary {artistId:score}
				- topk is the size of the putput prediction
				- lr is the learning rate used for gradient descent 
				- regFactor is the regularization factor 
		"""
		### We fisrt normalize our values
		mean =0# np.mean([x[1] for x in queryVec.items()])
		for artistId in queryVec:
			queryVec[artistId] = float(queryVec[artistId])#
			
		### We perform gradient descent computing the error on the values given by the query
		projection = np.random.normal(1./math.sqrt(self.numFeatures), 1., self.numFeatures) #the projection of queryVec on u, estimated using gradient descent
		artist_score = queryVec.items()
		error=[]
		
		for iter in xrange(numIter):
			rnd.shuffle(artist_score)
			for artistId, targetScore in artist_score:
				err = targetScore - np.dot(projection,self.Q.transpose())[artistId]
				for k in xrange(self.numFeatures):
					projection[k] += lr*(err*self.Q[artistId,k]-regFactor*projection[k])
			s = []
			for aId in queryVec:
				s.append((queryVec[aId] - np.dot(projection,self.Q.transpose())[aId])**2)
			error.append(sum(s))


		plt.plot(error)
		plt.show()

		### using the projection of the query on u obtained by gradient descent we compute the estimated ratings
		reconstruction = np.dot(self.Q,projection)

		#projection = np.dot(self.Q.transpose(),[queryVec[idx] if idx in queryVec else 0. for idx in xrange(self.numArtists)])
		#reconstruction = np.dot(self.Q,projection)

		### we sort the ratings
		for artistId in queryVec: #we are not interested in prediction artists from the query 
			reconstruction[artistId] = 0.
		sortedArtistsId = np.argsort(reconstruction)[::-1]
		return sortedArtistsId[:topk],[reconstruction[val] for val in sortedArtistsId[:topk]]


if __name__ == "__main__":

	DATAPATH = "/home/mat/Documents/Git/GroovesharkRecommender/Recommender/Model_Based_approach/MF/saved_model/Q.npy"
	INV_LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/invLexicon.json"
	ARTISTS_MEAN_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/artistsMean.json"
	ARTISTS_PC_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/artistsPlaylistCount.json"

	INV_LEXICON = None
	ARTISTS_MEAN = None
	ARTISTS_PC = None

	with open(INV_LEXICON_PATH,'r') as jsonStream:
		INV_LEXICON = json.load(jsonStream)

	with open(ARTISTS_MEAN_PATH,'r') as jsonStream:
		ARTISTS_MEAN = json.load(jsonStream)['meanTable']

	with open(ARTISTS_PC_PATH,'r') as jsonStream:
		ARTISTS_PC = json.load(jsonStream)['countTable']

	QUERY = {39:5,2291:3,3929:3,289:5,262:1,1249:5} 

	print "###############################"
	print "##           QUERY           ##"
	print "###############################"
	for artistId in QUERY:
		print INV_LEXICON[str(artistId)],":",QUERY[artistId]

	rec = recommender(DATAPATH)
	answer,scores = rec.query(QUERY,ARTISTS_MEAN,ARTISTS_PC,20)

	print "###############################"
	print "##           answer          ##"
	print "###############################"
	for idx,artistId in enumerate(answer):
		print INV_LEXICON[str(artistId)],scores[idx]


