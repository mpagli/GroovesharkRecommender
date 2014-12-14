### Neighbor Based (NB) methods

NB methods consists in generating a recommendation as a weighted sum of playlists. The weights are most of the time a  measure of similarity between two users. In our case we have implemented a NB based recommender using the Jaccard and cosine similarity functions. The dataset we are using is made of 225413 playlists representing a total of 18729 artists.  


#### Jaccard similarity

The distance between playlist 1 and playlist 2 is expressed as the number of common elements normalized by the total number of elements in the two playlists:

			distance(p1, p2) = |p1 and p2| / |p1 or p2|

This distance function is interesting since it allows us to represent our playlists as sets, and operations on sets are very fast. 

For the following query: 

		artist name              count
		Lynyrd Skynyrd :           1
		Black Sabbath :            5
		Metallica :                5
		Iron Maiden :              3
		Jimi Hendrix :             5
		John Zorn :                3

In less than 4 seconds we got the following result:

		AC/DC  -> score: 150.355243357
		The Rolling Stones  -> score: 126.360771896
		Led Zeppelin  -> score: 117.300193042
		Guns N' Roses  -> score: 81.639725444
		The Who  -> score: 79.7989605225
		Creedence Clearwater Revival  -> score: 76.6355286885
		Nirvana  -> score: 62.2115772376
		The Doors  -> score: 61.73692011
		Eminem  -> score: 57.5714829112
		Linkin Park  -> score: 54.9079759605
		KISS  -> score: 52.6185734061
		U2  -> score: 51.4509084966
		Red Hot Chili Peppers  -> score: 51.3250838118
		Pearl Jam  -> score: 49.9318465331
		Aerosmith  -> score: 49.1124159019
		Queen  -> score: 48.371449128
		3 Doors Down  -> score: 46.1337253397
		Van Halen  -> score: 45.4068857315
		Muse  -> score: 45.1682646791
		Avenged Sevenfold  -> score: 44.6333900434

The prediction make sense but we are loosing a lot of information by not taking into account the count for each artist.

Another query:

		artist name              count
		John Zorn                 10
		Miles Davis               5
		John Coltrane             5
		Dave Brubeck              5

The answer:

		Charlie Parker  -> score: 31.3094380485
		Ella Fitzgerald  -> score: 29.1503391168
		Dizzy Gillespie  -> score: 28.2956190421
		Louis Armstrong  -> score: 25.3322921427
		Nina Simone  -> score: 22.7630826223
		Billie Holiday  -> score: 19.8798315973
		Charles Mingus  -> score: 15.2918357882
		Frank Sinatra  -> score: 13.9267440369
		Herbie Hancock  -> score: 13.5979623451
		Thelonious Monk  -> score: 13.2730493526
		Duke Ellington  -> score: 12.6895588826
		Sarah Vaughan  -> score: 12.5770839843
		Chet Baker  -> score: 11.5892304413
		Ray Charles  -> score: 11.1173170419
		Wes Montgomery  -> score: 9.90956098887
		Stan Getz  -> score: 9.42304125402
		Sonny Rollins  -> score: 9.33186448065
		Count Basie  -> score: 9.01532862228
		Dexter Gordon  -> score: 8.58517780686
		Jaco Pastorius  -> score: 8.13281013055


#### Cosine distance

The cosine distance is a measure of the angle between two playlists. Each playlist is represented by a vector of size the number of artists such that:

		vector[artist_index] = artist_count

		distance(vector1, vector2) = dot(vector1, vector2) / (|vector1|.|vector2|)

This distance function is much more computationally intensive than the jaccard distance. For the same query, it took about 1 minute to generate the following recommendation:

		AC/DC  -> score: 1329.56966799
		The Rolling Stones  -> score: 1240.04095922
		Led Zeppelin  -> score: 1033.70724799
		Guns N' Roses  -> score: 855.03324657
		Eminem  -> score: 746.002462549
		Nirvana  -> score: 721.337591404
		Red Hot Chili Peppers  -> score: 668.467438024
		Pantera  -> score: 638.734972786
		Linkin Park  -> score: 625.538186942
		Megadeth  -> score: 581.325566406
		The Doors  -> score: 570.431461998
		Creedence Clearwater Revival  -> score: 556.196485283
		Pearl Jam  -> score: 545.958220073
		Queen  -> score: 514.075850768
		The Who  -> score: 475.888291788
		KISS  -> score: 466.134774114
		Green Day  -> score: 465.781693294
		Bob Marley  -> score: 418.612336552
		Avenged Sevenfold  -> score: 417.944424298
		Aerosmith  -> score: 415.62205419

Second query: 

		artist name              count
		John Zorn                 10
		Miles Davis               5
		John Coltrane             5
		Dave Brubeck              5

The answer:

		Charlie Parker  -> score: 356.697928386
		Ella Fitzgerald  -> score: 211.015861807
		Dizzy Gillespie  -> score: 189.937832745
		Chick Corea  -> score: 165.560423275
		Thelonious Monk  -> score: 165.257592645
		Louis Armstrong  -> score: 145.87691475
		Diana Krall  -> score: 108.416705237
		Wes Montgomery  -> score: 98.4319800445
		Sonny Rollins  -> score: 94.4518639806
		Billie Holiday  -> score: 94.1663643776
		Toni Braxton  -> score: 87.9434437472
		Frank Sinatra  -> score: 84.6856552336
		Cannonball Adderley  -> score: 81.1409596409
		Herbie Hancock  -> score: 80.4209328755
		Nina Simone  -> score: 79.4447239795
		Jaco Pastorius  -> score: 78.5589209583
		The Klezmatics  -> score: 78.3525652245
		Charles Mingus  -> score: 77.6245787026
		Marvin Gaye  -> score: 75.8424676536
		Buena Vista Social Club  -> score: 73.482722401



