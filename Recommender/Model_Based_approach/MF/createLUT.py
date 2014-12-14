#!/usr/bin/python

import json

LEXICON_PATH = "/home/mat/Documents/Git/backup/pygrooveshark/src/playlists/lexicon.json"
LEXICON = None

with open(LEXICON_PATH,'r') as jsonStream:
	LEXICON = json.load(jsonStream)

fstream = open("artistLUT.txt",'w') 
for artistId in LEXICON["artists"]:
	try:
		if LEXICON["artists"][artistId]["count"] > 300:
			fstream.write(LEXICON["artists"][artistId]["name"]+": "+str(artistId)+"\n")
	except:
		pass
fstream.close()