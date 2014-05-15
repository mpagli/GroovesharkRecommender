#!/usr/bin/python

import subprocess
import sys
import json
import time
from grooveshark import Client
from multiprocessing.dummy import Pool as ThreadPool

client = Client()
client.init()

if len(sys.argv) < 3 or False in (sys.argv[1].isdigit(),sys.argv[2].isdigit()):
	print "Usage : ./pullPlaylists <startIdx> <endIdx>"
	sys.exit(2)

idStart = int(sys.argv[1])
idStop  = int(sys.argv[2]) 

outPath = "./playlists/"

count = 0 	#total number of playlists saved

def getPlaylistFromId(id_):
	try :
		query = client.playlist(str(id_))
		if id_ == query['PlaylistID']: 	#query['PlaylistID']=0 when the playlist doesn't exists 
			jsonPath = outPath + str(query['PlaylistID']) + ".json"
			with open(jsonPath,'w') as stream:
				json.dump(query, stream)
			print "\tPlaylist " + str(id_) + " saved as " + jsonPath
			return 1
		else:
			return 0
	except :
		return 0	

pool = ThreadPool(25)
results = pool.map(getPlaylistFromId, range(max(1,idStart),idStop+1))
pool.close()
pool.join()

print str(sum(results)) + " have been pulled in " + outPath

