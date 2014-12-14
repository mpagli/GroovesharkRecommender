#!/usr/bin/python

import numpy as np
from rsvd import RSVD
import json
import os
import random as rnd 
import sys
import math

MEAN_TABLE = None
ARTISTS_PLAYLIST_COUNT = None

class artistData(object):
	def __init__(self,artistIdx):
		self.artistIdx = artistIdx
		self.occurrences = [] 
		self.mean = None

	def addOccurence(self,count):
		self.occurrences.append(count)

	def getMean(self,denominator=None):
		if denominator == None:
			denominator = len(self.occurrences)
		if self.mean==None:
			self.mean = float(sum(self.occurrences))/denominator
		MEAN_TABLE[self.artistIdx] = self.mean
		return self.mean

def formatData(corpus,lexicon):
	"""return a numpy record array from the original corpus, lexicon and mapping LUT"""
	outCorpus = []
	for playListIdx,playListId in enumerate(corpus):

		numSong = float(sum([x[1] for x in corpus[playListId]]))

		for artistId,count in corpus[playListId]:

			artistIdx = lexicon[artistId].artistIdx
			count -= lexicon[artistId].getMean()	#subtract the mean corresponding to the popularity of the artist across the corpus 
			count = float(count)/numSong		#normalize the score so the length of the playlist does not matters
			#count -= np.mean([x[1] for x in corpus[playListId]])
			#count /= ARTISTS_PLAYLIST_COUNT[artistIdx]

			outCorpus.append((playListIdx,artistIdx,count))
	return outCorpus

def splitDataset(corpus):
	"""from a list of tuples, return 3 chunk: training/validation and test"""
	Idx = range(len(corpus))
	rnd.shuffle(Idx)
	trainStop = int(len(corpus)*0.77)
	validStop = int(len(corpus)*0.999)
	trainIdx = Idx[:trainStop]
	validIdx = Idx[trainStop:validStop]
	testIdx  = Idx[validStop:]
	trainSet = [corpus[i] for i in trainIdx]#np.array([corpus[i] for i in trainIdx],dtype=[('f0', np.uint16),('f1', np.uint32),('f2',np.float32)])
	validSet = np.array([corpus[i] for i in validIdx],dtype=[('f0', np.uint16),('f1', np.uint32),('f2',np.float32)])
	testSet  = np.array([corpus[i] for i in testIdx ],dtype=[('f0', np.uint16),('f1', np.uint32),('f2',np.float32)])
	return trainSet,validSet,testSet

def removeSmallPlaylists(corpus,lexicon,threshold_artistCount,threshold_artistPop):
	"""remove all the playlist that contain less than threshold_artistCount different artists with a popularity superior to threshold_artistPop"""
	toDelete = []
	newLexicon  = {}	#map the artistid to [0,numArtists]
	newInvLexicon = {}	#map [0,numArtists] to artistName
	for playListId in corpus:
		acc = {}
		for songId in corpus[playListId]:
			artistId = lexicon["songs"][songId]["artistId"]
			if lexicon["artists"][artistId]["count"] < threshold_artistPop:
				continue
			if artistId in acc:
				acc[artistId] += 1
			else:
				acc[artistId] =  1
		if len(acc) < threshold_artistCount:
			toDelete.append(playListId)
		else:
			corpus[playListId] = acc.items()
			for artistId in acc:
				if artistId not in newLexicon:
					newLexicon[artistId] = artistData(len(newLexicon))
					newLexicon[artistId].addOccurence(acc[artistId])
					newInvLexicon[newLexicon[artistId].artistIdx] = lexicon["artists"][artistId]["name"]
				else :
					newLexicon[artistId].addOccurence(acc[artistId])
				artistIdx = newLexicon[artistId].artistIdx
				if artistIdx in ARTISTS_PLAYLIST_COUNT:
					ARTISTS_PLAYLIST_COUNT[artistIdx] += 1
				else :
					ARTISTS_PLAYLIST_COUNT[artistIdx] = 1
	for playListId in toDelete:
		del corpus[playListId]
	return corpus,newLexicon,newInvLexicon

if __name__ == "__main__":

	CORPUS_PATH  = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/corpus.json"
	LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/lexicon.json"

	CORPUS = None	#contain all the playlists as lists of (artistsId,count)
	LEXICON = None	#map all the artistsId to an unique index in [0,numArtists]
	INV_LEXICON = None	#map [0,numArtists] to the artists name

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

	ARTISTS_PLAYLIST_COUNT = {}

	CORPUS,LEXICON,INV_LEXICON = removeSmallPlaylists(CORPUS,LEXICON,threshold_artistCount=5,threshold_artistPop=500)#threshold_artistCount=3,threshold_artistPop=100)

	with open("/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/invLexicon.json", 'w') as outfile:
		json.dump(INV_LEXICON, outfile)

	MEAN_TABLE = [None]*len(LEXICON)

	f_corpus = formatData(CORPUS,LEXICON)

	with open("/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/artistsMean.json", 'w') as outfile:
		json.dump({"meanTable":MEAN_TABLE}, outfile)

	with open("/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/artistsPlaylistCount.json", 'w') as outfile:
		json.dump({"countTable":ARTISTS_PLAYLIST_COUNT}, outfile)

	print "\nNumber of playlists:",len(CORPUS),"\nNumber of artists:",len(LEXICON)

	### Split the dataset
	print "\nGenerating the training/validation and test set ...",
	sys.stdout.flush()
	trainSet,validSet,testSet = splitDataset(f_corpus)
	print "ok"

	### train our MF model
	import RISMF as MF
	mf = MF.RISMF(trainSet,numItems=len(LEXICON),numUsers=len(CORPUS)+1,weightDecay=0.0002,learningRate=0.0001,numFeatures=25)#WD 0.2

	try:
		mf.train(numSamples=1,maxEpoch=60,maxNumAfterBest = 5)
	except:
		print "Training aborted..."

	mf.save("/home/mat/Documents/Git/GroovesharkRecommender/Recommender/Model_Based_approach/MF/saved_model/")
	
	"""prototype: train(factors,ratingsArray,dims,probeArray=None,\
                  maxEpochs=100,minImprovement=0.000001,\
                  learnRate=0.001,regularization=0.011,\
                  randomize=False, randomNoise=0.005)"""
	"""
	model = RSVD.train(25,trainSet,(len(LEXICON),len(CORPUS)),validSet,learnRate=0.0005,regularization=0.005)
	model.save("./")
	"""