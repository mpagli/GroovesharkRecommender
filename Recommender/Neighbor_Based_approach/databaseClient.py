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
			with open(file,'r') as jsonStream:
				try:
					jsonData = json.load(jsonStream)
					if 'Songs' in jsonData and len(jsonData['Songs']) > 7:
						collection.insert({key: jsonData[key] for key in fieldsToKeep}) 
				except ValueError:
					print "Warning: Broken json",file

	def dropCollection(self,collectionName):
		""""""
		self.db[collectionName].drop()

	def dropDatabase(self):
		self.db.connection.drop_database("db")

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
		
		
