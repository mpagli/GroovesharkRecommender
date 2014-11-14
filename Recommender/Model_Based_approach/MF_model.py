#!/usr/bin/python

import numpy as np
from rsvd import RSVD, rating_t
import json
import os
import random as rnd 
import sys

def formatData(corpus,lexicon,playlists_LUT,artists_LUT):
	"""return a numpy record array from the original corpus, lexicon and maping LUT"""
	outCorpus = []
	for playListId in corpus:
		acc = {}
		for songId in corpus[playListId]:
			artistId = lexicon["songs"][songId]["artistId"]
			if artistId in acc:
				acc[artistId] += 1
			else:
				acc[artistId] =  1
		for artistId,counts in acc.items():
			outCorpus.append((artists_LUT[artistId],playlists_LUT[playListId],counts))
	return outCorpus#np.recarray(outCorpus,dtype=[('artistId', uint16),('playListId', uint32),,('rating':float)])

def splitDataset(corpus):
	"""from a list of tuples, return 3 chunk: training/validation and test"""
	Idx = range(len(corpus))
	rnd.shuffle(Idx)
	trainStop = int(len(corpus)*0.70)
	validStop = int(len(corpus)*0.95)
	trainIdx = Idx[:trainStop]
	validIdx = Idx[trainStop:validStop]
	testIdx  = Idx[validStop:]
	trainSet = np.array([corpus[i] for i in trainIdx],dtype=[('f0', np.uint16),('f1', np.uint32),('f2',np.float32)])
	validSet = np.array([corpus[i] for i in validIdx],dtype=[('f0', np.uint16),('f1', np.uint32),('f2',np.float32)])
	testSet  = np.array([corpus[i] for i in testIdx ],dtype=[('f0', np.uint16),('f1', np.uint32),('f2',np.float32)])
	return trainSet,validSet,testSet

def removeSmallPlaylists(corpus,lexicon,threshold_artistCount,threshold_artistPop):
	"""remove all the playlist that contain less than threshold_artistCount different artists with a popularity superior to threshold_artistPop"""
	toDelete = []
	for playListId in corpus:
		acc = {}
		newSongList = []
		for songId in corpus[playListId]:
			artistId = lexicon["songs"][songId]["artistId"]
			if lexicon["artists"][artistId]["count"] < threshold_artistPop:
				continue
			if artistId in acc:
				acc[artistId] += 1
			else:
				acc[artistId] =  1
			newSongList.append(songId)
		corpus[playListId] = newSongList
		if len(acc) < threshold_artistCount:
			toDelete.append(playListId)
	for playListId in toDelete:
		del corpus[playListId]
	return corpus

if __name__ == "__main__":

	CORPUS_PATH  = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/corpus.json"
	LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/lexicon.json"
	PLAYLISTS_LUT_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/playlists_lut.json"
	ARTISTS_LUT_PATH   = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/artists_lut.json"
	INV_ARTISTS_LUT_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/inv_playlists_lut.json"

	CORPUS = None	#contain all the playlists as lists of songId
	LEXICON = None	#contain all the data for each artist/song
	PLAYLISTS_LUT = None	#map the grooveshark playlistId to a local playlistId 
	ARTISTS_LUT = None	#map the grooveshark artistId to a local artistId
	INV_ARTISTS_LUT = None	#map the local artistId to the grooveshark one

	### LOADING THE DATA
	print "\nLoading corpus from "+CORPUS_PATH+" ...",
	sys.stdout.flush()
	with open(CORPUS_PATH,'r') as jsonStream:
		CORPUS = json.load(jsonStream)
	print "ok"
	print "\nLoading lexicon from "+LEXICON_PATH+" ...",
	sys.stdout.flush()
	with open(LEXICON_PATH,'r') as jsonStream:
		LEXICON = json.load(jsonStream)
	print "ok"

	CORPUS = removeSmallPlaylists(CORPUS,LEXICON,threshold_artistCount=20,threshold_artistPop=300)

	### FORMATING THE DATA AS TUPLES: (artistId,playlistId,count)
	print "\nFormating the dataset ..."
	sys.stdout.flush()
	if not os.path.isfile(PLAYLISTS_LUT_PATH): #we then need to create the LUT
		print "\nCreating PLAYLISTS_LUT ...",
		sys.stdout.flush()
		PLAYLISTS_LUT   = {}
		for playListId in CORPUS:
			if playListId not in PLAYLISTS_LUT:
				PLAYLISTS_LUT[playListId] = len(PLAYLISTS_LUT)
		with open(PLAYLISTS_LUT_PATH, 'w') as outfile:
  			json.dump(PLAYLISTS_LUT, outfile)
  		print "ok"
	else:
		with open(PLAYLISTS_LUT_PATH,'r') as jsonStream:
			PLAYLISTS_LUT = json.load(jsonStream)


	if not os.path.isfile(ARTISTS_LUT_PATH): 
		print "\nCreating ARTISTS_LUT ...",
		sys.stdout.flush()
		ARTISTS_LUT   = {}
		INV_ARTISTS_LUT = {}
		for playListId in CORPUS:
			for songId in CORPUS[playListId]:
				artistId = LEXICON["songs"][songId]["artistId"]
				if artistId not in ARTISTS_LUT:
					ARTISTS_LUT[artistId] = len(ARTISTS_LUT)+1
					INV_ARTISTS_LUT[len(ARTISTS_LUT)+1] = artistId
		print "ok"
		with open(ARTISTS_LUT_PATH, 'w') as outfile:
  			json.dump(ARTISTS_LUT, outfile)
  		with open(INV_ARTISTS_LUT_PATH, 'w') as outfile:
  			json.dump(INV_ARTISTS_LUT, outfile)
	else:
		with open(ARTISTS_LUT_PATH,'r') as jsonStream:
			ARTISTS_LUT = json.load(jsonStream)
		with open(INV_ARTISTS_LUT_PATH,'r') as jsonStream:
			INV_ARTISTS_LUT = json.load(jsonStream)

	f_corpus = formatData(CORPUS,LEXICON,PLAYLISTS_LUT,ARTISTS_LUT)
	CORPUS.clear() #no need to keep corpus in memory 
	print "ok"

	print "\nNumber of playlists:",len(PLAYLISTS_LUT),"\nNumber of artists:",len(ARTISTS_LUT)

	### Split the dataset
	print "\nGenerating the training/validation and test set ...",
	sys.stdout.flush()
	trainSet,validSet,testSet = splitDataset(f_corpus)
	print "ok"

	### train our MF model
	"""prototype: train(factors,ratingsArray,dims,probeArray=None,\
                  maxEpochs=100,minImprovement=0.000001,\
                  learnRate=0.001,regularization=0.011,\
                  randomize=False, randomNoise=0.005)"""
	model = RSVD.train(25,trainSet,(len(ARTISTS_LUT),len(PLAYLISTS_LUT)),validSet,learnRate=0.00001,regularization=0.001,minImprovement=0.0000001,maxEpochs=10000)


