#!/usr/bin/python

import databaseClient as dbClient

class vectorsBase(object):
	"""base class for the vectors container, links all the vectors to the proper collection"""

	db = None
	collectionName = None
	dataPath = None

	def __init__(self,dataClient,dataPath,collectionName,fieldsToKeep):
		"""modify static attributes"""
		vectorsBase.collectionName = collectionName		
		vectorsBase.dataPath = dataPath	
		vectorsBase.db = dataClient
		vectorsBase.db.addCollection(dataPath,collectionName,fieldsToKeep) #load all the data in dataPath
	

		
		
