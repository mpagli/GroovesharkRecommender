#!/usr/bin/python

import vectorsBase as vb 
import userVectors as uv 
import collaborativeFiltering as cf 
import databaseClient as dbClient

import sys
import os 
import numpy as np 

if __name__ == "__main__":

	dataPath       = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/playlists/"	#folder that contain the json data
	collectionName = "playlists"					#name of the collection in our DB
	fieldsToKeep   = ['Songs']						#we only keep the data for the field "Songs"

	field = 'Songs'									
	subFields = ['ArtistID','ArtistName']			#we extract the artist Id and name subfields in the list of songs 

	lexicon = {}	#maps labels to an idx 
	invLexicon = {} #maps idx to labels

	print "\n\n#### Loading the data from",dataPath
	myDB = dbClient.databaseClient()	
	#myDB.dropDatabase()						#simple interface to pymongo
	#myDB.dropCollection(collectionName)
	vb.vectorsBase(myDB,dataPath,collectionName,fieldsToKeep)	#load the data into myDB with the collection name "collectionName", links the userVectors class to this collection
	myUV = uv.userVectors(field,subFields)	
	print "####",myDB.getCount(collectionName),"documents loaded"

	idx = 0
	processedUV = []#[None]*myDB.getCount(collectionName) #filtered version of user-vectors to be send to the collaborative filtering module

	print "\n\nProcessing the user vectors:",
	for userIdx,userVector in enumerate(myUV.data):	#for each user-vector in myDB
		for item in userVector:						#for each song in the playlist
			artistID   = item[0]
			artistName = item[1]
			if artistID not in lexicon:
				lexicon[artistID] = idx
				invLexicon[idx]   = artistName
				idx += 1
		s =  set([lexicon[x[0]] for x in userVector])
		if len(s) > 7:
			processedUV.append(s)#[userIdx]
	print len(processedUV),"users processed.",len(lexicon),"labels."

	stream = open("LUT.txt",'w')
	for key,val in invLexicon.items(): #display the correspondence between artist names and key so we can build a fake userVector
		try:
			stream.write(str(key)+"\t"+str(val)+"\n")
		except ValueError:
			pass
	stream.close()

	#For this implementation the uservectors are just sets of artist idx : {1,40,2,6 ...}
	currentUser = set([1453,92275,1434,1437,1564,1809,1849,1908,2194,2473,2476,2481,2506,2890,2895,2902])
	print "\n################ Current User:\n"
	for item in currentUser:
		print invLexicon[item]
	print "\n\nComputing similarities"
	recommender = cf.collaborativeFiltering(processedUV,lexicon) #create sparse rating matrix
	recommendation = recommender.getFullRecommendationVector(currentUser,'jaccard')
	for item in currentUser:	#we are not interested in recommending already known artists
		recommendation[item] = 0.

	topK = list(np.argsort(recommendation)[-50:]) #extract the top K recommendation
	topK.reverse()
	print "\n################ Recommendations:\n"
	for rec in topK:
		print invLexicon[rec],recommendation[rec]


