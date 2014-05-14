#!/usr/bin/python

import subprocess
import sys
import json
import time
from grooveshark import Client

client = Client()
client.init()

if len(sys.argv) < 3 or False in (sys.argv[1].isdigit(),sys.argv[2].isdigit()):
	print "Usage : ./pullPlaylists <startIdx> <endIdx>"
	sys.exit(2)

idStart = int(sys.argv[1])
idStop  = int(sys.argv[2]) 

outPath = "./playlists/"

count = 0 	#total number of playlists saved

for id_ in xrange(max(1,idStart),idStop+1):
	try :
		query = client.playlist(str(id_))
		if id_ == query['PlaylistID']: 	#query['PlaylistID']=0 when the playlist doesn't exists 
			jsonPath = outPath + str(query['PlaylistID']) + ".json"
			with open(jsonPath,'w') as stream:
				json.dump(query, stream)
			print "\tPlaylist " + str(id_) + " saved as " + jsonPath
			count += 1
	except :
		time.sleep(60)

print str(count) + " have been pulled in " + outPath

