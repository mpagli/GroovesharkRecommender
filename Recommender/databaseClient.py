#!/usr/bin/python

import pymongo
import json 
import glob

class databaseClient(object):
	"""class to load the database, filter data"""

	def __init__(self):
		"""Constructor: create a mongoDB client"""
		self.client = pymongo.MongoClient()
		self.db = self.client.db
		self.collectionFolders = {}

	def addCollection(self,dataPath,collectionName,fieldsToKeep):
		"""load all json file in datapath folder into a db.collectionName collection"""
		self.collectionFolders[collectionName] = dataPath
		collection = self.db[collectionName]
		files = glob.iglob(dataPath+'*.json') #iglob == generator
		for file in files:
			with open(filepath,'r') as jsonStream:
				jsonData = json.load(jsonStream)
				collection.insert({key: jsonData[key] for key in fieldsToKeep}) 

	def get_id(self,_id,collectionName):
		"""return the element of id _id"""
		return self.db[collectionName].find({"_id" : _id})
	
	def getAll(self,collectionName):
		"""return an iterator over all the collection"""
		return self.db[collectionName].find()

	def getCount(self,collectionName):
		"""return the number of elements in the collection"""
		return self.db[collectionName].count()

	#missing: displayStat / other display tools ... 
		
		
