#!/usr/bin/python

import vectorsBase as vb 
import userVectors as uv 
import collaborativeFiltering as cf 
import databaseClient as dbClient

import sys
import os 
import numpy as np 

if __name__ == "__main__":

	dataPath       = "./../Data/samples_playlists/"	#folder that contain the json data
	collectionName = "playlists"					#name of the collection in our DB
	fieldsToKeep   = ['Songs']						#we only keep the data for the field "Songs"

	field = 'Songs'									
	subFields = ['ArtistID','ArtistName']			#we extract the artist Id and name subfields in the list of songs 

	lexicon = {}	#maps labels to an idx 
	invLexicon = {} #maps idx to labels

	myDB = dbClient.databaseClient()							#simple interface to pymongo
	vb.vectorsBase(myDB,dataPath,collectionName,fieldsToKeep)	#load the data into myDB with the collection name "collectionName", links the userVectors class to this collection
	myUV = uv.userVectors(field,subFields)	

	idx = 0
	processedUV = [None]*myDB.getCount(collectionName) #filtered version of user-vectors to be send to the collaborative filtering module

	for userIdx,userVector in enumerate(myUV.data):	#for each user-vector in myDB
		for item in userVector:						#for each song in the playlist
			artistID   = item[0]
			artistName = item[1]
			if artistID not in lexicon:
				lexicon[artistID] = idx
				invLexicon[idx]   = artistName
				idx += 1
		processedUV[userIdx] = set([lexicon[x[0]] for x in userVector])

	for key,val in invLexicon.items(): #display the correspondence between artist names and key so we can build a fake userVector
		print key,"\t",val

	#For this implementation the uservectors are just sets of artist idx : {1,40,2,6 ...}
	currentUser = set([472,470,279,238,89,471,453])
	print "\n################ Current User:\n"
	for item in currentUser:
		print invLexicon[item]
	recommender = cf.collaborativeFiltering(processedUV,lexicon) #create sparse rating matrix
	recommendation = recommender.getFullRecommendationVector(currentUser,'jaccard')

	topK = np.argsort(recommendation)[-10:] #extract the top K recommendation
	print "\n################ Recommendation:\n"
	for rec in topK:
		print invLexicon[rec]


