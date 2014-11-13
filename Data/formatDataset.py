#!/usr/bin/python

import json
import sys
import glob

if __name__=="__main__":
	"""Loads all the json playlist situated in the dataPaths folders, save formated data in 2 different files: corpus.json contain the playlists, lexicon.json contains some LUT to link the song/artists Ids to the name..."""

	if len(sys.argv) < 2:
		print "usage: ./formatDataset.py <dataPath1> <dataPath2> ... "
		sys.exit(2)

	dataPaths = sys.argv[1:]

	corpus  = {}
	lexicon = {}
	lexicon["artists"] = {}
	lexicon["songs"]   = {}	 

	for dataPath in dataPaths:
		files = glob.iglob(dataPath+'*.json') #return a generator to all the files' name in datapath 
		for file in files:
			with open(file,'r') as jsonStream:
				try:
					jsonData = json.load(jsonStream)
					if 'Songs' in jsonData and len(jsonData['Songs']) > 5:	#We discard small playlists
						playlistId =  jsonData["PlaylistID"]

						if playlistId in corpus: 
							print "Warning: playlist ",playlistId," already in corpus"
							continue

						playlist = [None]*len(jsonData["Songs"])

						for idx,song in enumerate(jsonData["Songs"]):
							songId    = song["SongID"]
							songName  = song["Name"]
							artistId  = song["ArtistID"]
							artistName= song["ArtistName"]
							popularity= song["Popularity"]
															
							if songId in lexicon["songs"]:		#Update lexicon["songs"]
								lexicon["songs"][songId]["count"] += 1
							else:
								lexicon["songs"][songId] = {}
								lexicon["songs"][songId]["count"] = 1
								lexicon["songs"][songId]["name"]  = songName
								lexicon["songs"][songId]["popularity"] = popularity
								lexicon["songs"][songId]["artistId"]   = artistId

							if artistId in lexicon["artists"]:	#Update lexicon["artists"]
								lexicon["artists"][artistId]["count"] += 1
							else:
								lexicon["artists"][artistId] = {}
								lexicon["artists"][artistId]["count"] = 1
								lexicon["artists"][artistId]["name"]  = artistName
							
							playlist[idx] = songId
							
						corpus[playlistId] = playlist

				except ValueError:
					print "Warning: Broken json ",file

	with open('corpus.json', 'w') as outfile:
  		json.dump(corpus, outfile)
	with open('lexicon.json', 'w') as outfile:
  		json.dump(lexicon, outfile)
	
