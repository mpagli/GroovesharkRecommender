#!/usr/bin/python

import vectorsBase as vb

class userVectors(vb.vectorsBase):
	"""class to clean the data and return a generator of userVectors"""

	def __init__(self,field,subFields):
		""""""
		self.data = self.extractUserVectors(field,subFields)

	def extractUserVectors(self,field,subFields):
		""""""
		for dataItem in self.db.getAll(self.collectionName):
			yield [[subItem[sf] for sf in subFields] for subItem in dataItem[field]]

"""vbase(params)
   for users in UserVectors(): ..."""
