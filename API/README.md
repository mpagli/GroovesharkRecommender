### How to make this work with pyGrooveshark

pyGrooveshark not only pull the playlists/collections ... but also parse it. Since we are just interested in getting the raw json format we can bypass the parsing.

You can do this by modifying this : 

    return self._parse_playlist(playlist)
    
for this: 

    return playlist
    
in grooveshark/__init__.py
