#!/usr/bin/python

import recommendationModule as rm
import numpy as np
import scipy.sparse as sps

class collaborativeFiltering(rm.recommendationModule):
	"""class to perform collab filtering"""

	def __init__(self,userVectors,lexicon):
		"""userVectors are lists of list (or lists of set) of features such as artist name/ songId ... each user is a 1D list. lexicon is a LUT that maps all labels to a unique id that will be used as index of that feature in the arrays"""
		self.userVectors = userVectors
		self.lexicon = lexicon
		self.numItems = len(lexicon)
		self.numUsers = len(userVectors)
		self.ratingMatrix = sps.coo_matrix((self.numUsers,self.numItems)) #sparse matrix
		for idx,user in enumerate(self.userVectors):
			rows = [idx for x in user]
			cols = [x for x in user]
			data = [1 for x in user]
			self.ratingMatrix = self.ratingMatrix + sps.coo_matrix((data,(rows,cols)),shape=(self.numUsers,self.numItems))
		self.ratingMatrix = sps.csc_matrix(self.ratingMatrix)

	def getFullRecommendationVector(self,currentUser,distanceFunc):
		"""return an array containing the similarity score obtained as the product Su*R, Su being the vector of similarity between the current and all other users(computed using the selected distanceFunc) and R the ratings matrix"""
		if distanceFunc == "jaccard":
			"""assume that the userVectors are sets, simply check the number of elements in common for the two users being compared normalized by the size of the union"""
			Su = [None]*len(xrange(self.numUsers))
			for idx,user in enumerate(self.userVectors):
				user_user_sim = sum([1. for x in currentUser if x in user]) / len(user|currentUser)
				Su[idx] = user_user_sim
			Su = np.array(Su) 
			return Su*self.ratingMatrix
		elif distanceFunc == "cosine":
			"""assume the uservectros are all numpy arrays of the same length"""
			pass

	
