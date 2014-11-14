#!/usr/bin/python

import numpy as np 
import matplotlib.pyplot as plt 
import pylab as P
import sys
import json

if __name__ == "__main__":

	CORPUS_PATH  = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/corpus.json"
	LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/lexicon.json"

	CORPUS = None	#contain all the playlists as lists of songId
	LEXICON = None	#contain all the data for each artist/song

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

	hist_popularity_artists = [None]*len(LEXICON["artists"])
	for idx,artistId in enumerate(LEXICON["artists"]):
		hist_popularity_artists[idx] = min(LEXICON["artists"][artistId]['count'],500)

	hist_num_artists_playlists = [None]*len(CORPUS)
	for idx,playlistId in enumerate(CORPUS):
		acc = {}
		for songId in CORPUS[playlistId]:
			artistId = LEXICON["songs"][songId]["artistId"]
			if artistId not in acc:
				acc[artistId] = 1
		hist_num_artists_playlists[idx] = min(len(acc),100)


	n, bins, patches = P.hist(hist_popularity_artists, 500, histtype='stepfilled')
	P.show()

	n, bins, patches = P.hist(hist_num_artists_playlists, 100, histtype='stepfilled')
	P.show()
